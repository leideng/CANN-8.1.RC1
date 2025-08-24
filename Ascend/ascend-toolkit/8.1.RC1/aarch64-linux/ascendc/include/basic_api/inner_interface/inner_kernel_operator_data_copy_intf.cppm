/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file inner_kernel_operator_data_copy_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_DATA_COPY_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_DATA_COPY_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_process_lock.h"

#include "kernel_check.h"

#include "impl/kernel_operator_data_copy_base_impl.h"

namespace AscendC {
/* **************************************************************************************************
 * DataCopy                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataCopy Level 0
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcGlobal input GlobalTensor
 * @param [in] intriParams.blockCount number of blocks
 * @param [in] intriParams.blockLen Length of blocks
 * @param [in] intriParams.srcStride src block stride
 * @param [in] intriParams.dstStride dst block stride
 */
template <typename T>
__aicore__ inline void __inout_pipe__(MTE2) DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
    const DataCopyParams& repeatParams)
{
    using PrimType = PrimT<T>;
    const Hardware dstHWPos = GetPhyType((TPosition)dstLocal.GetPosition());
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncDataCopy(dstLocal, srcGlobal, repeatParams, "DataCopy from GlobalTensor to LocalTensor")) {
        ASCENDC_REPORT_CHECK_ERROR("DataCopy from GlobalTensor to LocalTensor", KernelFuncType::NONE_MODE);
    }
    ASCENDC_REPORT_OVERFLOW_MEM(CheckDataCopyTensorSizeOverflow(dstLocal, srcGlobal, repeatParams));
#endif
    if (dstHWPos == Hardware::UB) {
        // gm -> ub
        DataCopyGM2UBImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__gm__ PrimType*)srcGlobal.GetPhyAddr(),
            repeatParams);
    } else if (dstHWPos == Hardware::L1) {
        // gm -> l1
        DataCopyGM2L1Impl((__cbuf__ PrimType*)dstLocal.GetPhyAddr(), (__gm__ PrimType*)srcGlobal.GetPhyAddr(),
            repeatParams);
    } else {
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "A1 / B1 / C1 / VECIN",
            "DataCopy from GlobalTensor to LocalTensor with DataCopyParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}

__aicore__ inline void CheckNd2NzParams(Nd2NzParams params, const __gm__ char *msg)
{
    constexpr uint16_t ND2NZ_LIMIT = 16384; // nValue, dstNzC0Stride, dstNzNStride must be in range [0, 16384]
    ASCENDC_CHECK_VALUE_RANGE(params.ndNum, 0, UINT12_MAX, "ndNum", msg);
    ASCENDC_CHECK_VALUE_RANGE(params.nValue, 0, ND2NZ_LIMIT, "nValue", msg);
    ASCENDC_CHECK_VALUE_RANGE(params.dstNzC0Stride, 0, ND2NZ_LIMIT, "dstNzC0Stride", msg);
    ASCENDC_CHECK_VALUE_RANGE(params.dstNzNStride, 0, ND2NZ_LIMIT, "dstNzNStride", msg);
}

/*
 * @ingroup DataCopy Level 0
 * @brief format transform(such as nd2nz) during data load from OUT to L1
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcGlobal input GlobalTensor
 * @param [in] intriParams.ndNum nd number of data to be moved
 * @param [in] intriParams.nValue n value
 * @param [in] intriParams.dValue d value in unit of element
 * @param [in] intriParams.srcNdMatrixStride stride between nd matrixs at source ND matrix in unit of element
 * @param [in] intriParams.srcDValue SRC_D value in unit of element
 * @param [in] intriParams.dstNzC0Stride stride of nz between 2 C0 in L1 in unit of C0_size
 * @param [in] intriParams.dstNzNStride stride of n between 2 C0 in L1
 * @param [in] intriParams.dstNzMatrixStride DST_nz_matrix_stride in L1 in unit of element
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
    const Nd2NzParams& intriParams)
{
    CheckNd2NzParams(intriParams, "DataCopy with Nd2NzParams");
    using PrimType = PrimT<T>;
    const Hardware dstHWPos = GetPhyType((TPosition)dstLocal.GetPosition());
    ASCENDC_REPORT_OVERFLOW_MEM(CheckDataCopyTensorSizeOverflow(dstLocal, srcGlobal, intriParams));
    if (dstHWPos == Hardware::L1) {
        // gm -> l1
        DataCopyGM2L1ND2NZImpl((__cbuf__ PrimType*)dstLocal.GetPhyAddr(), (__gm__ PrimType*)srcGlobal.GetPhyAddr(),
            intriParams);
    } else if (dstHWPos == Hardware::UB) {
        DataCopyGM2UBND2NZImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__gm__ PrimType*)srcGlobal.GetPhyAddr(),
            intriParams);
    } else {
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "A1 / B1 / VECIN",
            "DataCopy from GlobalTensor to LocalTensor with Nd2NzParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}

/*
 * @ingroup DataCopy Level 0
 * @brief format transform(such as nd2nz) during data load from UB to L1(Only TSCM)
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] intriParams.ndNum nd number of data to be moved, onlyc can be 1
 * @param [in] intriParams.nValue n value
 * @param [in] intriParams.dValue d value in unit of element
 * @param [in] intriParams.srcNdMatrixStride stride between nd matrixs at source ND matrix in unit of element
 * @param [in] intriParams.srcDValue SRC_D value in unit of element
 * @param [in] intriParams.dstNzC0Stride stride of nz between 2 C0 in L1 in unit of C0_size
 * @param [in] intriParams.dstNzNStride stride of n between 2 C0 in L1
 * @param [in] intriParams.dstNzMatrixStride DST_nz_matrix_stride in L1 in unit of element
 */
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const Nd2NzParams &intriParams)
{
    CheckNd2NzParams(intriParams, "DataCopy with Nd2NzParams");
    using PrimType = PrimT<T>;
    CheckTensorPos<T>(srcLocal, Hardware::UB, "srcLocal", "VECIN / VECCALC / VECOUT",
        "DataCopy from LocalTensor to LocalTensor with Nd2NzParams");
    CheckTensorPos<T>(dstLocal, Hardware::L1, "dstLocal", "TSCM",
        "DataCopy from LocalTensor to LocalTensor with Nd2NzParams");
    ASCENDC_REPORT_OVERFLOW_MEM(CheckDataCopyTensorSizeOverflow(dstLocal, srcLocal, intriParams));
    DataCopyUB2L1ND2NZImpl((__cbuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        intriParams);
}

/*
 * @ingroup DataCopy Level 0
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstGlobal output GlobalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] intriParams.blockCount number of blocks
 * @param [in] intriParams.blockLen Length of blocks
 * @param [in] intriParams.srcStride src block stride
 * @param [in] intriParams.dstStride dst block stride
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const DataCopyParams& repeatParams)
{
    using PrimType = PrimT<T>;
    const Hardware srcHWPos = GetPhyType((TPosition)srcLocal.GetPosition());
#ifdef ASCENDC_CPU_DEBUG
    if (!CheckFuncDataCopy(dstGlobal, srcLocal, repeatParams, "DataCopy from LocalTensor to GlobalTensor")) {
        ASCENDC_REPORT_CHECK_ERROR("DataCopy from LocalTensor to GlobalTensor", KernelFuncType::NONE_MODE);
    }
    ASCENDC_REPORT_OVERFLOW_MEM(CheckDataCopyTensorSizeOverflow(dstGlobal, srcLocal, repeatParams));
    bool isUsedProcessLock = false;
    if (g_isAtomic == true) {
        ProcessLock::GetProcessLock()->Write();
        isUsedProcessLock = true;
    }
#endif // ASCENDC_CPU_DEBUG
    if (srcHWPos == Hardware::UB) {
        // ub -> gm
        DataCopyUB2GMImpl((__gm__ PrimType*)dstGlobal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
            repeatParams);
    } else if (srcHWPos == Hardware::L1) {
        // l1 -> gm
        DataCopyL12GMImpl((__gm__ PrimType*)dstGlobal.GetPhyAddr(), (__cbuf__ PrimType*)srcLocal.GetPhyAddr(),
            repeatParams);
    } else {
#if __CCE_AICORE__ == 200
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "A1 / B1 / CO2 / VECOUT",
            "DataCopy from LocalTensor to GlobalTensor with DataCopyParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
#else
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "A1 / B1 / VECOUT",
            "DataCopy from LocalTensor to GlobalTensor with DataCopyParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
#endif
    }

#ifdef ASCENDC_CPU_DEBUG
    if (isUsedProcessLock == true) {
        isUsedProcessLock = false;
        ProcessLock::GetProcessLock()->Unlock();
    }
#endif // ASCENDC_CPU_DEBUG
}

/*
 * @ingroup DataCopy Level 0
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] intriParams.blockCount number of blocks
 * @param [in] intriParams.blockLen Length of blocks
 * @param [in] intriParams.srcStride src block stride
 * @param [in] intriParams.dstStride dst block stride
 */
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const DataCopyParams &repeatParams)
{
    const Hardware dstHWPos = GetPhyType((TPosition)dstLocal.GetPosition());
    const Hardware srcHWPos = GetPhyType((TPosition)srcLocal.GetPosition());

    ASCENDC_REPORT_OVERFLOW_MEM(CheckDataCopyTensorSizeOverflow(dstLocal, srcLocal, repeatParams));
    if (srcHWPos == Hardware::UB) {
        if (dstHWPos == Hardware::UB) {
            // ub -> ub
#if ASCENDC_CPU_DEBUG
            if (!CheckFuncDataCopy(dstLocal, srcLocal, repeatParams, "DataCopy from LocalTensor to LocalTensor")) {
                ASCENDC_REPORT_CHECK_ERROR("DataCopy from LocalTensor to LocalTensor", KernelFuncType::NONE_MODE);
            }
#endif
            DataCopyUB2UBIntf(dstLocal, srcLocal, repeatParams);
        } else if (dstHWPos == Hardware::L1) {
            // ub -> l1
            DataCopyUB2L1Intf(dstLocal, srcLocal, repeatParams);
        } else {
#if __CCE_AICORE__ == 220
            ASCENDC_CHECK_TPOSITION(false, "dstLocal", "VECCALC / VECOUT / TSCM",
                "DataCopy from LocalTensor(VECIN / VECCALC / VECOUT) to LocalTensor with DataCopyParams",
                ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
#else
            ASCENDC_CHECK_TPOSITION(false, "dstLocal", "VECCALC / VECOUT / A1 / B1",
                "DataCopy from LocalTensor(VECIN / VECCALC / VECOUT) to LocalTensor with DataCopyParams",
                ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
#endif
        }
    } else if (srcHWPos == Hardware::L1) {
        if (dstHWPos == Hardware::UB) {
            // l1 -> ub
            DataCopyL12UBIntf(dstLocal, srcLocal, repeatParams);
        } else if (dstHWPos == Hardware::BIAS) {
            CheckTensorAlign<T>(dstLocal, 64, "dstLocal", "DataCopy from C1 to C2");            // 64B align
            CheckTensorAlign<T>(srcLocal, ONE_BLK_SIZE, "srcLocal", "DataCopy from C1 to C2");  // 32B align
            DataCopyL12BTIntf(dstLocal, srcLocal, repeatParams);
#if __CCE_AICORE__ >= 220
        } else if (dstHWPos == Hardware::FIXBUF) {
            CheckTensorAlign<T>(dstLocal, 128, "dstLocal", "DataCopy from A1 / B1 / C1 to C2PIPE2GM");  // 128B align
            CheckTensorAlign<T>(srcLocal, ONE_BLK_SIZE, "srcLocal", "DataCopy from C1 to C2");          // 32B align
            DataCopyL12FBIntf(dstLocal, srcLocal, repeatParams);
#endif
        } else {
            ASCENDC_CHECK_TPOSITION(false, "dstLocal", "C2 / C2PIPE2GM",
                "DataCopy from LocalTensor(A1 / B1 / C1) to LocalTensor with DataCopyParams",
                ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
        }
    } else {
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "VECIN / VECCALC / VECOUT / A1 / B1 / C1",
            "DataCopy from LocalTensor to LocalTensor with DataCopyParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
    }
}

/*
 * @ingroup DataCopy Level 0
 * @brief datacopy from L1 to bt, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] intriParams.blockCount number of blocks
 * @param [in] intriParams.blockLen Length of blocks
 * @param [in] intriParams.srcStride src block stride
 * @param [in] intriParams.dstStride dst block stride
 */
template <typename dst_T, typename src_T>
__aicore__ inline void DataCopy(const LocalTensor<dst_T> &dstLocal, const LocalTensor<src_T> &srcLocal,
    const DataCopyParams &repeatParams)
{
    using PrimDstType = PrimT<dst_T>;
    using PrimSrcType = PrimT<src_T>;
    const Hardware dstHWPos = GetPhyType((TPosition)dstLocal.GetPosition());
    const Hardware srcHWPos = GetPhyType((TPosition)srcLocal.GetPosition());

    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow<dst_T, src_T>(dstLocal, srcLocal, repeatParams)));
    if (srcHWPos == Hardware::L1) {
        if (dstHWPos == Hardware::BIAS) {
            // l1 -> bt
            CheckTensorAlign<dst_T>(dstLocal, 64, "dstLocal", "DataCopy from C1 to C2");            // 64B align
            CheckTensorAlign<src_T>(srcLocal, ONE_BLK_SIZE, "srcLocal", "DataCopy from C1 to C2");  // 32B align
            if constexpr (IsSameType<PrimDstType, PrimSrcType>::value) {
                DataCopyL12BTImpl((uint64_t)dstLocal.GetPhyAddr(), (__cbuf__ PrimSrcType*)srcLocal.GetPhyAddr(),
                    (uint16_t)0, repeatParams);
            } else if constexpr (IsSameType<PrimDstType, float>::value && IsSameType<PrimSrcType, half>::value) {
                DataCopyL12BTImpl((uint64_t)dstLocal.GetPhyAddr(), (__cbuf__ half *)srcLocal.GetPhyAddr(), (uint16_t)1,
                    repeatParams);
            } else {
                ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in DataCopy from C1 to C2, "
                    "current api support dtype combination is src_T = dst_T or src: half, dst: float.");});
            }
        } else {
            ASCENDC_CHECK_TPOSITION(false, "dstLocal", "C2",
                "DataCopy from LocalTensor to LocalTensor with dst_T / src_T",
                ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
        }
    } else {
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "C1",
            "DataCopy from LocalTensor to LocalTensor with dst_T / src_T",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
    }
}

/*
 * @ingroup Copy Level 0
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstStride dst block stride
 * @param [in] intriParams.srcStride src block stride
 * @param [in] intriParams.dstRepeatSize dst repeat stride
 * @param [in] intriParams.srcRepeatSize src repeat stride
 */
// Copy::Level 0 - mask bit mode
template <typename T, bool IsSetMask>
__aicore__ inline __inout_pipe__(V) void Copy(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const uint64_t mask[], const uint8_t repeatTimes, const CopyRepeatParams &repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(IsSetMask);
    if (!CheckFuncCopy(dstLocal, srcLocal, mask, repeatTimes, repeatParams, "Copy")) {
        ASCENDC_REPORT_CHECK_ERROR("Copy", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    CopyImpl<PrimType, IsSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}

// Copy::Level 0 - mask count mode
template <typename T, bool IsSetMask>
__aicore__ inline __inout_pipe__(V) void Copy(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const uint64_t mask, const uint8_t repeatTimes, const CopyRepeatParams &repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(IsSetMask);
    if (!CheckFuncCopy(dstLocal, srcLocal, mask, repeatTimes, repeatParams, "Copy")) {
        ASCENDC_REPORT_CHECK_ERROR("Copy", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    CopyImpl<PrimType, IsSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}

/*
 * @ingroup DataCopy Level 1
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcGlobal input GlobalTensor
 * @param [in] SliceInfo dstSliceInfo[] ub
 * @param [in] SliceInfo srcSliceInfo[] gm
 * @param [in] dimValue dim value also for length for dstSliceInfo[] and srcSliceInfo[]
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGlobal,
    const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[], const uint32_t dimValue)
{
    using PrimType = PrimT<T>;
    static_assert(IsSameType<PrimType, T>::value, "TensorTrait is not supported by DataCopy with SliceInfo!");
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncDataCopySlice(dstLocal, srcGlobal, dstSliceInfo, srcSliceInfo, dimValue, "DataCopy with SliceInfo")) {
        ASCENDC_REPORT_CHECK_ERROR("DataCopy with SliceInfo", KernelFuncType::NONE_MODE);
    }
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstLocal, srcGlobal, dstSliceInfo, srcSliceInfo, dimValue)));
#endif
    uint32_t srcStartIndex = 0;
    uint32_t dstStartIndex = 0;
    uint32_t srcOffsetListSize = 0;
    uint32_t dstOffsetListSize = 0;
    uint32_t srcShapeInfo[K_MAX_SHAPE_DIM];
    uint32_t dstShapeInfo[K_MAX_SHAPE_DIM];
    bool useShapeValue = !(srcSliceInfo[0].shapeValue == 0);
    for (int i = 0; i < dimValue; i++) {
        srcShapeInfo[i] = useShapeValue ? srcSliceInfo[i].shapeValue : srcGlobal.GetShapeInfo().shape[i];
        dstShapeInfo[i] = useShapeValue ? dstSliceInfo[i].shapeValue : dstLocal.GetShapeInfo().shape[i];
    }

    srcStartIndex = DataCopyGetPhyStartIndex(srcSliceInfo, srcShapeInfo, dimValue);
    dstStartIndex = DataCopyGetPhyStartIndex(dstSliceInfo, dstShapeInfo, dimValue);
    uint32_t srcOffsetList[MAX_SLICE_SIZE];
    uint32_t dstOffsetList[MAX_SLICE_SIZE];
    DataCopyGetOffsetList(srcSliceInfo, srcShapeInfo, dimValue, &srcOffsetListSize, srcOffsetList);
    DataCopyGetOffsetList(dstSliceInfo, dstShapeInfo, dimValue, &dstOffsetListSize, dstOffsetList);
    struct DataCopyParams repeatParams;
    repeatParams.blockLen = srcSliceInfo[0].burstLen;
    uint32_t oneSliceLen = srcSliceInfo[0].burstLen * AscendCUtils::GetC0Count(sizeof(T)) + srcSliceInfo[0].stride;
    repeatParams.blockCount =
        (srcSliceInfo[0].endIndex - srcSliceInfo[0].startIndex + 1 + srcSliceInfo[0].stride) / oneSliceLen;
    repeatParams.dstStride = dstSliceInfo[0].stride * sizeof(T) / AscendCUtils::GetC0Size();

    if ((srcSliceInfo[0].stride * sizeof(T)) % AscendCUtils::GetC0Size() == 0) {
        repeatParams.srcStride = srcSliceInfo[0].stride * sizeof(T) / AscendCUtils::GetC0Size();
        for (uint32_t i = 0; i < srcOffsetListSize; i++) {
            DataCopyGM2UBImpl((__ubuf__ T *)dstLocal.GetPhyAddr() + dstStartIndex + dstOffsetList[i],
                (__gm__ T *)srcGlobal.GetPhyAddr() + srcStartIndex + srcOffsetList[i], repeatParams);
        }
    } else {
        repeatParams.srcStride = srcSliceInfo[0].stride * sizeof(T);
        for (uint32_t i = 0; i < srcOffsetListSize; i++) {
            DataCopySliceGm2UBImpl((__ubuf__ T *)dstLocal.GetPhyAddr() + dstStartIndex + dstOffsetList[i],
                (__gm__ T *)srcGlobal.GetPhyAddr() + srcStartIndex + srcOffsetList[i], repeatParams);
        }
    }
}

/*
 * @ingroup DataCopy Level 1
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcGlobal input GlobalTensor
 * @param [in] SliceInfo dstSliceInfo[] gm
 * @param [in] SliceInfo srcSliceInfo[] ub
 * @param [in] dimValue dim value also for length for dstSliceInfo[] and srcSliceInfo[]
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopy(const GlobalTensor<T> &dstGlobal, const LocalTensor<T> &srcLocal,
    const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[], const uint32_t dimValue)
{
    using PrimType = PrimT<T>;
    static_assert(IsSameType<PrimType, T>::value, "TensorTrait is not supported by DataCopy with SliceInfo!");
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncDataCopySlice(dstGlobal, srcLocal, dstSliceInfo, srcSliceInfo, dimValue, "DataCopy with SliceInfo")) {
        ASCENDC_REPORT_CHECK_ERROR("DataCopy with SliceInfo", KernelFuncType::NONE_MODE);
    }
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstGlobal, srcLocal, dstSliceInfo, srcSliceInfo, dimValue)));
#endif
    uint32_t srcStartIndex = 0;
    uint32_t dstStartIndex = 0;
    uint32_t srcOffsetListSize = 0;
    uint32_t dstOffsetListSize = 0;
    uint32_t srcShapeInfo[K_MAX_SHAPE_DIM];
    uint32_t dstShapeInfo[K_MAX_SHAPE_DIM];
    bool useShapeValue = !(srcSliceInfo[0].shapeValue == 0);
    for (int i = 0; i < dimValue; i++) {
        srcShapeInfo[i] = useShapeValue ? srcSliceInfo[i].shapeValue : srcLocal.GetShapeInfo().shape[i];
        dstShapeInfo[i] = useShapeValue ? dstSliceInfo[i].shapeValue : dstGlobal.GetShapeInfo().shape[i];
    }

    srcStartIndex = DataCopyGetPhyStartIndex(srcSliceInfo, srcShapeInfo, dimValue);
    dstStartIndex = DataCopyGetPhyStartIndex(dstSliceInfo, dstShapeInfo, dimValue);
    uint32_t dstOffsetList[MAX_SLICE_SIZE];
    uint32_t srcOffsetList[MAX_SLICE_SIZE];
    DataCopyGetOffsetList(srcSliceInfo, srcShapeInfo, dimValue, &srcOffsetListSize, srcOffsetList);
    DataCopyGetOffsetList(dstSliceInfo, dstShapeInfo, dimValue, &dstOffsetListSize, dstOffsetList);

    struct DataCopyParams repeatParams;
    repeatParams.blockLen = srcSliceInfo[0].burstLen;
    uint32_t oneSliceLen = srcSliceInfo[0].burstLen * AscendCUtils::GetC0Count(sizeof(T)) + srcSliceInfo[0].stride;
    repeatParams.blockCount =
        (srcSliceInfo[0].endIndex - srcSliceInfo[0].startIndex + 1 + srcSliceInfo[0].stride) / oneSliceLen;
    repeatParams.srcStride = srcSliceInfo[0].stride * sizeof(T) / AscendCUtils::GetC0Size();

    if ((dstSliceInfo[0].stride * sizeof(T)) % AscendCUtils::GetC0Size() == 0) {
        repeatParams.dstStride = dstSliceInfo[0].stride * sizeof(T) / AscendCUtils::GetC0Size();
        for (uint32_t i = 0; i < srcOffsetListSize; i++) {
            DataCopyUB2GMImpl((__gm__ T *)dstGlobal.GetPhyAddr() + dstStartIndex + dstOffsetList[i],
                (__ubuf__ T *)srcLocal.GetPhyAddr() + srcStartIndex + srcOffsetList[i], repeatParams);
        }
    } else {
        repeatParams.dstStride = dstSliceInfo[0].stride * sizeof(T);
        for (uint32_t i = 0; i < srcOffsetListSize; i++) {
            DataCopySliceUB2GMImpl((__gm__ T *)dstGlobal.GetPhyAddr() + dstStartIndex + dstOffsetList[i],
                (__ubuf__ T *)srcLocal.GetPhyAddr() + srcStartIndex + srcOffsetList[i], repeatParams);
        }
    }
}

/*
 * @ingroup DataCopy Level 2
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcGlobal input GlobalTensor
 * @param [in] calCount Number of operands
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
    const uint32_t calCount)
{
    using PrimType = PrimT<T>;
    ASCENDC_ASSERT((calCount % AscendCUtils::GetC0Count(sizeof(PrimType)) == 0), { KERNEL_LOG(KERNEL_ERROR, "Failed to "
        "check calCount value in DataCopy from GlobalTensor to LocalTensor, calCount * sizeof(T) must be 32B align, "
        "current calCount value is %u. In NPU mode, no error is reported. The value is rounded down by 32B.",
        calCount); });
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstLocal, srcGlobal, calCount)));
    struct DataCopyParams repeatParams;
    repeatParams.blockLen = calCount / AscendCUtils::GetC0Count(sizeof(PrimType));
    DataCopy(dstLocal, srcGlobal, repeatParams);
}

/*
 * @ingroup DataCopy Level 2
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstGlobal output GlobalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calCount Number of operands
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const uint32_t calCount)
{
    using PrimType = PrimT<T>;
    ASCENDC_ASSERT((calCount % AscendCUtils::GetC0Count(sizeof(PrimType)) == 0), { KERNEL_LOG(KERNEL_ERROR, "Failed to "
        "check calCount value in DataCopy from LocalTensor to GlobalTensor, calCount * sizeof(T) must be 32B align, "
        "current calCount value is %u. In NPU mode, no error is reported. The value is rounded down by 32B.",
        calCount); });
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstGlobal, srcLocal, calCount)));
    struct DataCopyParams repeatParams;
    repeatParams.blockLen = calCount / AscendCUtils::GetC0Count(sizeof(PrimType));
    DataCopy(dstGlobal, srcLocal, repeatParams);
}

/*
 * @ingroup DataCopy Level 2
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calCount Number of operands
 */
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, const uint32_t calCount)
{
    using PrimType = PrimT<T>;
    ASCENDC_ASSERT((calCount % AscendCUtils::GetC0Count(sizeof(PrimType)) == 0), { KERNEL_LOG(KERNEL_ERROR, "Failed to "
        "check calCount value in DataCopy from LocalTensor to LocalTensor, calCount * sizeof(T) must be 32B align, "
        "current calCount value is %u. In NPU mode, no error is reported. The value is rounded down by 32B.",
        calCount); });
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstLocal, srcLocal, calCount)));
    struct DataCopyParams repeatParams;

    const Hardware dstHWPos = GetPhyType((TPosition)dstLocal.GetPosition());
    const Hardware srcHWPos = GetPhyType((TPosition)srcLocal.GetPosition());
    if (srcHWPos != Hardware::L1) {  // UB -> UB, UB -> L1
        repeatParams.blockLen = calCount / AscendCUtils::GetC0Count(sizeof(PrimType));
    } else {                         // L1 -> UB, L1 -> BT, L1 -> FB
        if (dstHWPos == Hardware::UB) {
            repeatParams.blockLen = calCount / AscendCUtils::GetC0Count(sizeof(PrimType));
        } else if (dstHWPos == Hardware::BIAS) {
            repeatParams.blockLen = calCount / (64 / sizeof(PrimType));   // BT blockLen is in unit of 64B
        } else if (dstHWPos == Hardware::FIXBUF) {
            repeatParams.blockLen = calCount / (128 / sizeof(PrimType));  // FB blockLen is in unit of 128B
        }
    }
    DataCopy(dstLocal, srcLocal, repeatParams);
}

__aicore__ inline void CheckNz2NdParams(const Nz2NdParamsFull& params)
{
    constexpr uint16_t NZ2ND_LIMIT = 8192;      // nValue, dstNzC0Stride, dstNzNStride must be in range [0, 16384]
    ASCENDC_CHECK_VALUE_RANGE(params.ndNum, 0, UINT12_MAX, "ndNum", "DataCopy with Nz2NdParamsFull");
    ASCENDC_CHECK_VALUE_RANGE(params.nValue, 1, NZ2ND_LIMIT, "nValue", "DataCopy with Nz2NdParamsFull");
    ASCENDC_CHECK_VALUE_RANGE(params.dValue, 1, NZ2ND_LIMIT, "dValue", "DataCopy with Nz2NdParamsFull");
    ASCENDC_CHECK_VALUE_RANGE(params.srcNdMatrixStride, 1, VALUE_512, "srcNdMatrixStride", "DataCopy with Nz2NdParamsFull");
    ASCENDC_CHECK_VALUE_RANGE(params.srcNStride, 0, UINT12_MAX, "srcNStride", "DataCopy with Nz2NdParamsFull");
}

/*
 * @ingroup DataCopy Level 2
 * @brief datacopy from src to dst, nz2nd, applicable to simulated cube data(such as data from l0c, 16*16)
 * @param [out] dstGlobal output GlobalTensor
 * @param [in] srcLocal input LocalTensor
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const Nz2NdParamsFull& intriParams)
{
    CheckNz2NdParams(intriParams);
    using PrimType = PrimT<T>;
#ifdef ASCENDC_CPU_DEBUG
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstGlobal, srcLocal, intriParams)));
    bool isUsedProcessLock = false;
    if (g_isAtomic == true) {
        ProcessLock::GetProcessLock()->Write();
        isUsedProcessLock = true;
    }
#endif // ASCENDC_CPU_DEBUG
    const Hardware srcHWPos = GetPhyType((TPosition)srcLocal.GetPosition());
    if (srcHWPos != Hardware::UB) {
#if __CCE_AICORE__ == 200
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "VECOUT / CO2", "DataCopy with Nz2NdParamsFull",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
#else
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "VECOUT", "DataCopy with Nz2NdParamsFull",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
#endif
    }
    DataCopyUB2GMNZ2NDImpl((__gm__ PrimType*)dstGlobal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        intriParams);
#ifdef ASCENDC_CPU_DEBUG
    if (isUsedProcessLock == true) {
        isUsedProcessLock = false;
        ProcessLock::GetProcessLock()->Unlock();
    }
#endif // ASCENDC_CPU_DEBUG
}

/* **************************************************************************************************
 * DataCopy Enhanced                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataCopy
 * @brief datacopy from src to dst, applicable to cube data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcGlobal input GlobalTensor
 * @param [in] intriParams.blockCount number of blocks
 * @param [in] intriParams.blockLen Length of blocks
 * @param [in] intriParams.srcStride src block stride
 * @param [in] intriParams.dstStride dst block stride
 * @param [in] enhancedParams.blockMode Basic fractal of data movement
 * @param [in] enhancedParams.deqScale Auxiliary parameters for path accuracy conversion
 * @param [in] enhancedParams.deqValue size of convert with path precision
 * @param [in] enhancedParams.sidStoreMode Multiplex input
 * @param [in] enhancedParams.isRelu Configure whether Relu can be performed along the circuit
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
    const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams)
{
    using PrimType = PrimT<T>;
    const Hardware dstHWPos = GetPhyType((TPosition)dstLocal.GetPosition());
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstLocal, srcGlobal, intriParams, enhancedParams)));

    if (dstHWPos == Hardware::UB) {
        // gm -> ub
        DataCopyGM2UBImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__gm__ PrimType*)srcGlobal.GetPhyAddr(),
            intriParams);
    } else if (dstHWPos == Hardware::L1) {
        // gm -> l1
        DataCopyGM2L1Impl((__cbuf__ PrimType*)dstLocal.GetPhyAddr(), (__gm__ PrimType*)srcGlobal.GetPhyAddr(),
            intriParams);
    } else {
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "A1 / B1 / VECIN",
            "DataCopy from GlobalTensor to LocalTensor with DataCopyEnhancedParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}

template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams)
{
    using PrimType = PrimT<T>;
    const Hardware srcHWPos = GetPhyType((TPosition)srcLocal.GetPosition());
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstGlobal, srcLocal, intriParams, enhancedParams)));

    if (srcHWPos == Hardware::UB) {
        // ub -> gm
        DataCopyUB2GMImpl((__gm__ PrimType*)dstGlobal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
            intriParams);
    } else if (srcHWPos == Hardware::L1) {
        // l1 -> gm
        DataCopyL12GMImpl((__gm__ PrimType*)dstGlobal.GetPhyAddr(), (__cbuf__ PrimType*)srcLocal.GetPhyAddr(),
            intriParams);
    } else {
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "A1 / B1 / VECOUT",
            "DataCopy from LocalTensor to GlobalTensor with DataCopyEnhancedParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
    }
}

template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    const Hardware dstHWPos = GetPhyType((TPosition)dstLocal.GetPosition());
    const Hardware srcHWPos = GetPhyType((TPosition)srcLocal.GetPosition());
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstLocal, srcLocal, intriParams, enhancedParams)));

    if (srcHWPos == Hardware::UB) {
        if (dstHWPos == Hardware::L1) {
            // ub -> l1
            DataCopyUB2L1Intf(dstLocal, srcLocal, intriParams);
        } else if (dstHWPos == Hardware::L0C) {
            // ub -> l0c
            DataCopyUB2L0CIntf(dstLocal, srcLocal, intriParams, enhancedParams);
        } else if (dstHWPos == Hardware::UB) {
            // ub -> ub
            DataCopyUB2UBIntf(dstLocal, srcLocal, intriParams);
        } else {
            ASCENDC_CHECK_TPOSITION(false, "dstLocal", "VECCALC / VECOUT / A1 / B1 / TSCM",
                "DataCopy from LocalTensor(VECIN / VECCALC / VECOUT) to LocalTensor with DataCopyEnhancedParams",
                ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
        }
    } else if (srcHWPos == Hardware::L1) {
        if (dstHWPos == Hardware::UB) {
            // l1 -> ub
            DataCopyL12UBIntf(dstLocal, srcLocal, intriParams);
        } else if (dstHWPos == Hardware::L0C) {
            // l1 -> l0c
            DataCopyL12L0CIntf(dstLocal, srcLocal, intriParams, enhancedParams);
        } else {
            ASCENDC_CHECK_TPOSITION(false, "dstLocal", "CO1",
                "DataCopy from LocalTensor(A1 / B1) to LocalTensor with DataCopyEnhancedParams",
                ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
        }
    } else if (srcHWPos == Hardware::L0C) {
        if (dstHWPos == Hardware::UB) {
            // l0c -> ub
            DataCopyL0C2UBIntf(dstLocal, srcLocal, intriParams, enhancedParams);
        } else {
            ASCENDC_CHECK_TPOSITION(false, "dstLocal", "CO2",
                "DataCopy from LocalTensor(CO1) to LocalTensor with DataCopyEnhancedParams",
                ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
        }
    } else {
#if __CCE_AICORE__ == 200
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "VECIN / CO1",
            "DataCopy from LocalTensor to LocalTensor with DataCopyEnhancedParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
#else
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "VECIN",
            "DataCopy from LocalTensor to LocalTensor with DataCopyEnhancedParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
#endif
    }
}

template <typename T, typename U>
__aicore__ inline void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
    const DataCopyCO12DstParams& intriParams)
{
    CheckTensorPos<U>(srcLocal, Hardware::L0C, "srcLocal", "CO1",
        "DataCopy from LocalTensor to LocalTensor with DataCopyCO12DstParams");
    CheckTensorPos<T>(dstLocal, Hardware::L1, "dstLocal", "A1",
        "DataCopy from LocalTensor to LocalTensor with DataCopyCO12DstParams");
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstLocal, srcLocal, intriParams)));
    // l0c -> l1
    DataCopyL0C2L1Impl((__cbuf__ PrimT<T>*)dstLocal.GetPhyAddr(), (__cc__ PrimT<U>*)srcLocal.GetPhyAddr(), intriParams);
}

template <typename T, typename U>
__aicore__ inline void DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<U>& srcLocal,
    const DataCopyCO12DstParams& intriParams)
{
    CheckTensorPos<U>(srcLocal, Hardware::L0C, "srcLocal", "CO1",
        "DataCopy from LocalTensor to GlobalTensor with DataCopyCO12DstParams");
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstGlobal, srcLocal, intriParams)));
    // l0c -> gm
    DataCopyL0C2GMImpl((__gm__ PrimT<T>*)dstGlobal.GetPhyAddr(), (__cc__ PrimT<U>*)srcLocal.GetPhyAddr(), intriParams);
}

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
// float to bfloat16_t
template <typename T, typename U, typename std::enable_if<IsSameType<PrimT<T>, bfloat16_t>::value &&
    IsSameType<PrimT<U>, float>::value, bool>::type>
__aicore__ inline void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
    const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams)
{
    const Hardware dstHWPos = GetPhyType((TPosition)dstLocal.GetPosition());
    const Hardware srcHWPos = GetPhyType((TPosition)srcLocal.GetPosition());
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstLocal, srcLocal, intriParams, enhancedParams)));
    if (srcHWPos == Hardware::L1) {
        if (dstHWPos == Hardware::L0C) {
            // l1 -> l0c
            DataCopyL12L0CImpl((__cc__ PrimT<T>*)dstLocal.GetPhyAddr(), (__cbuf__ PrimT<U>*)srcLocal.GetPhyAddr(),
                intriParams, enhancedParams);
        } else {
            ASCENDC_CHECK_TPOSITION(false, "dstLocal", "CO1",
                "DataCopy from LocalTensor(U) to LocalTensor(T) with DataCopyEnhancedParams",
                ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
        }
    } else {
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "A1 / B1",
            "DataCopy from LocalTensor(U) to LocalTensor(T) with DataCopyEnhancedParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
    }
}
#endif

// float to half
template <typename T, typename U, typename std::enable_if<IsSameType<PrimT<T>, half>::value &&
    IsSameType<PrimT<U>, float>::value, bool>::type>
__aicore__ inline void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
    const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams)
{
    const Hardware dstHWPos = GetPhyType((TPosition)dstLocal.GetPosition());
    const Hardware srcHWPos = GetPhyType((TPosition)srcLocal.GetPosition());
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstLocal, srcLocal, intriParams, enhancedParams)));
    if (srcHWPos == Hardware::L1) {
        if (dstHWPos == Hardware::L0C) {
            // l1 -> l0c
            DataCopyL12L0CImpl((__cc__ half*)dstLocal.GetPhyAddr(), (__cbuf__ float*)srcLocal.GetPhyAddr(), intriParams,
                enhancedParams);
        } else {
            ASCENDC_CHECK_TPOSITION(false, "dstLocal", "CO1",
                "DataCopy from LocalTensor(A1/B1) to LocalTensor with DataCopyEnhancedParams",
                ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
        }
    } else if (srcHWPos == Hardware::L0C) {
        if (dstHWPos == Hardware::UB) {
            // l0c -> ub
            DataCopyL0C2UBImpl((__ubuf__ half*)dstLocal.GetPhyAddr(), (__cc__ float*)srcLocal.GetPhyAddr(), intriParams,
                enhancedParams);
        } else {
            ASCENDC_CHECK_TPOSITION(false, "dstLocal", "CO2",
                "DataCopy from LocalTensor(CO1) to LocalTensor with DataCopyEnhancedParams",
                ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
        }
    } else {
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "A1 / B1 / CO1",
            "DataCopy from LocalTensor(U) to LocalTensor(T) with DataCopyEnhancedParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}

template <typename T, typename U>
__aicore__ inline void CheckTensorL0C2UB(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal)
{
    CheckTensorPos<U>(srcLocal, Hardware::L0C, "srcLocal", "CO1",
        "DataCopy from LocalTensor(CO1) to LocalTensor(CO2) with DataCopyEnhancedParams");
    CheckTensorPos<T>(dstLocal, Hardware::UB, "dstLocal", "CO2",
        "DataCopy from LocalTensor(CO1) to LocalTensor(CO2) with DataCopyEnhancedParams");
}

// int32_t to half
template <typename T, typename U, typename std::enable_if<IsSameType<PrimT<T>, half>::value &&
    IsSameType<PrimT<U>, int32_t>::value, bool>::type>
__aicore__ inline __inout_pipe__(V) void DataCopy(const LocalTensor<T> &dstLocal, const LocalTensor<U> &srcLocal,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    CheckTensorL0C2UB(dstLocal, srcLocal);
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstLocal, srcLocal, intriParams, enhancedParams)));
    DataCopyL0C2UBImpl((__ubuf__ half*)dstLocal.GetPhyAddr(), (__cc__ int32_t*)srcLocal.GetPhyAddr(), intriParams,
        enhancedParams);
}

// int32_t to int16_t
template <typename T, typename U, typename std::enable_if<IsSameType<PrimT<T>, int16_t>::value &&
    IsSameType<PrimT<U>, int32_t>::value, bool>::type>
__aicore__ inline __inout_pipe__(V) void DataCopy(const LocalTensor<T> &dstLocal, const LocalTensor<U> &srcLocal,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    CheckTensorL0C2UB(dstLocal, srcLocal);
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstLocal, srcLocal, intriParams, enhancedParams)));
    DataCopyL0C2UBImpl((__ubuf__ int16_t*)dstLocal.GetPhyAddr(), (__cc__ int32_t*)srcLocal.GetPhyAddr(), intriParams,
        enhancedParams);
}

// int32_t to int8_t
template <typename T, typename U, typename std::enable_if<IsSameType<PrimT<T>, int8_t>::value &&
    IsSameType<PrimT<U>, int32_t>::value, bool>::type>
__aicore__ inline __inout_pipe__(V) void DataCopy(const LocalTensor<T> &dstLocal, const LocalTensor<U> &srcLocal,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    CheckTensorL0C2UB(dstLocal, srcLocal);
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstLocal, srcLocal, intriParams, enhancedParams)));
    DataCopyL0C2UBImpl((__ubuf__ int8_t*)dstLocal.GetPhyAddr(), (__cc__ int32_t*)srcLocal.GetPhyAddr(), intriParams,
        enhancedParams);
}

// int32_t to uint8_t
template <typename T, typename U, typename std::enable_if<IsSameType<PrimT<T>, uint8_t>::value &&
    IsSameType<PrimT<U>, int32_t>::value, bool>::type>
__aicore__ inline __inout_pipe__(V) void DataCopy(const LocalTensor<T> &dstLocal, const LocalTensor<U> &srcLocal,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    CheckTensorL0C2UB(dstLocal, srcLocal);
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstLocal, srcLocal, intriParams, enhancedParams)));
    DataCopyL0C2UBImpl((__ubuf__ uint8_t*)dstLocal.GetPhyAddr(), (__cc__ int32_t*)srcLocal.GetPhyAddr(), intriParams,
        enhancedParams);
}

// half to float
template <typename T, typename U, typename std::enable_if<IsSameType<PrimT<T>, float>::value &&
    IsSameType<PrimT<U>, half>::value, bool>::type>
__aicore__ inline __inout_pipe__(V) void DataCopy(const LocalTensor<T> &dstLocal, const LocalTensor<U> &srcLocal,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    CheckTensorPos<U>(srcLocal, Hardware::UB, "srcLocal", "CO2",
        "DataCopy from LocalTensor(CO2) to LocalTensor(CO1) with DataCopyEnhancedParams");
    CheckTensorPos<T>(dstLocal, Hardware::L0C, "dstLocal", "CO1",
        "DataCopy from LocalTensor(CO2) to LocalTensor(CO1) with DataCopyEnhancedParams");
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyTensorSizeOverflow(dstLocal, srcLocal, intriParams, enhancedParams)));
    DataCopyUB2L0CImpl((__cc__ float*)dstLocal.GetPhyAddr(), (__ubuf__ half*)srcLocal.GetPhyAddr(), intriParams,
        enhancedParams);
}


template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T> &dstLocal,
    const GlobalTensor<T> &srcGlobal, const DataCopyParams &dataCopyParams, const DataCopyPadParams &padParams)
{
    using PrimType = PrimT<T>;
    if ASCEND_IS_AIC {
        return;
    }
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncDataCopyPad(dstLocal, srcGlobal, dataCopyParams, padParams, "DataCopyPad from GM to VECIN/VECOUT")) {
        ASCENDC_REPORT_CHECK_ERROR("DataCopyPad from GM to VECIN / VECOUT", KernelFuncType::NONE_MODE);
    }
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyPadTensorSizeOverflow(dstLocal, srcGlobal, dataCopyParams, padParams)));
#endif
    const Hardware dstHWPos = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstHWPos == Hardware::UB) {
        DataCopyPadGm2UBImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__gm__ PrimType*)srcGlobal.GetPhyAddr(),
            dataCopyParams, padParams);
    } else if (dstHWPos == Hardware::L1) {
        DataCopyPadGM2L1Impl((__cbuf__ PrimType*)dstLocal.GetPhyAddr(), (__gm__ PrimType*)srcGlobal.GetPhyAddr(),
            dataCopyParams, padParams);
    } else {
#if defined(__DAV_M310__)
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "A1 / B1 / C1 / VECIN / VECOUT",
            "DataCopyPad from GlobalTensor to LocalTensor with DataCopyPadParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
#else
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "VECIN / VECOUT",
            "DataCopyPad from GlobalTensor to LocalTensor with DataCopyPadParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
#endif
    }
}

template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopyPad(const GlobalTensor<T> &dstGlobal,
    const LocalTensor<T> &srcLocal, const DataCopyParams &dataCopyParams)
{
    using PrimType = PrimT<T>;
    if ASCEND_IS_AIC {
        return;
    }
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyPadTensorSizeOverflow(dstGlobal, srcLocal, dataCopyParams)));
    const Hardware srcHWPos = GetPhyType((TPosition)srcLocal.GetPosition());
    if (srcHWPos == Hardware::UB) {
        DataCopyPadUB2GMImpl((__gm__ PrimType*)dstGlobal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
            dataCopyParams);
    } else if (srcHWPos == Hardware::L1) {
        DataCopyPadL12GMImpl((__gm__ PrimType*)dstGlobal.GetPhyAddr(), (__cbuf__ PrimType*)srcLocal.GetPhyAddr(),
            dataCopyParams);
    } else {
#if defined(__DAV_M310__)
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "A1 / B1 / C1 / VECIN / VECOUT",
            "DataCopyPad from LocalTensor to GlobalTensor with DataCopyParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
#else
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "VECIN / VECOUT",
            "DataCopyPad from LocalTensor to GlobalTensor with DataCopyParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
#endif
    }
}


template <typename T>
__aicore__ inline void DataCopyPad(const LocalTensor<T> &dstLocal,
    const LocalTensor<T> &srcLocal, const DataCopyParams &dataCopyParams, const Nd2NzParams &nd2nzParams)
{
    CheckNd2NzParams(nd2nzParams, "DataCopyPad with Nd2NzParams");
    using PrimType = PrimT<T>;
    CheckTensorPos<T>(dstLocal, Hardware::L1, "dstLocal", "TSCM", "DataCopyPad with Nd2NzParams");
    CheckTensorPos<T>(srcLocal, Hardware::UB, "srcLocal", "VECIN / VECOUT", "DataCopyPad with Nd2NzParams");
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyPadTensorSizeOverflow(dstLocal, srcLocal, dataCopyParams, nd2nzParams)));
    DataCopyPadUB2L1Impl((__cbuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        dataCopyParams, nd2nzParams);
}

// override DataCopyPad, use new param DataCopyExtParams
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T> &dstLocal,
    const GlobalTensor<T> &srcGlobal, const DataCopyExtParams &dataCopyParams, const DataCopyPadExtParams<T> &padParams)
{
    if ASCEND_IS_AIC {
        return;
    }
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncDataCopyPad(dstLocal, srcGlobal, dataCopyParams, padParams, "DataCopyPad from GM to VECIN/VECOUT")) {
        ASCENDC_REPORT_CHECK_ERROR("DataCopyPad from GM to VECIN / VECOUT", KernelFuncType::NONE_MODE);
    }
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyPadTensorSizeOverflow(dstLocal, srcGlobal, dataCopyParams, padParams)));
#endif
    const Hardware dstHWPos = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstHWPos == Hardware::UB) {
        DataCopyPadGm2UBImpl((__ubuf__ T*)dstLocal.GetPhyAddr(), (__gm__ T*)srcGlobal.GetPhyAddr(),
            dataCopyParams, padParams);
    } else if (dstHWPos == Hardware::L1) {
        DataCopyPadGM2L1Impl((__cbuf__ T*)dstLocal.GetPhyAddr(), (__gm__ T*)srcGlobal.GetPhyAddr(),
            dataCopyParams, padParams);
    } else {
#if defined(__DAV_M310__)
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "A1 / B1 / C1 / VECIN / VECOUT",
            "DataCopyPad from GM to VECIN/VECOUT",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
#else
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "VECIN / VECOUT",
            "DataCopyPad from GM to VECIN/VECOUT",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
#endif
    }
}

// override DataCopyPad, use new param DataCopyExtParams
// T use TensorTrait while U is primitive type
template <typename T, typename U, typename std::enable_if<IsSameType<PrimT<T>, U>::value &&
    (!IsSameType<T, U>::value), bool>::type>
__aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T> &dstLocal,
    const GlobalTensor<T> &srcGlobal, const DataCopyExtParams &dataCopyParams, const DataCopyPadExtParams<U> &padParams)
{
    using PrimType = PrimT<T>;
    if ASCEND_IS_AIC {
        return;
    }
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncDataCopyPad(dstLocal, srcGlobal, dataCopyParams, padParams, "DataCopyPad from GM to VECIN/VECOUT")) {
        ASCENDC_REPORT_CHECK_ERROR("DataCopyPad from GM to VECIN / VECOUT", KernelFuncType::NONE_MODE);
    }
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyPadTensorSizeOverflow(dstLocal, srcGlobal, dataCopyParams, padParams)));
#endif
    CheckTensorPos<T>(dstLocal, Hardware::UB, "dstLocal", "VECIN / VECOUT", "DataCopyPad from GM to VECIN / VECOUT");
    DataCopyPadGm2UBImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__gm__ PrimType*)srcGlobal.GetPhyAddr(),
        dataCopyParams, padParams);
}

template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopyPad(const GlobalTensor<T> &dstGlobal,
    const LocalTensor<T> &srcLocal, const DataCopyExtParams &dataCopyParams)
{
    using PrimType = PrimT<T>;
    if ASCEND_IS_AIC {
        return;
    }
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyPadTensorSizeOverflow(dstGlobal, srcLocal, dataCopyParams)));
    const Hardware srcHWPos = GetPhyType((TPosition)srcLocal.GetPosition());
    if (srcHWPos == Hardware::UB) {
        DataCopyPadUB2GMImpl((__gm__ PrimType*)dstGlobal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
            dataCopyParams);
    } else if (srcHWPos == Hardware::L1) {
        DataCopyPadL12GMImpl((__gm__ PrimType*)dstGlobal.GetPhyAddr(), (__cbuf__ PrimType*)srcLocal.GetPhyAddr(),
            dataCopyParams);
    } else {
#if defined(__DAV_M310__)
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "A1 / B1 / C1 / VECIN / VECOUT",
            "DataCopyPad from LocalTensor to GlobalTensor with DataCopyExtParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
#else
        ASCENDC_CHECK_TPOSITION(false, "srcLocal", "VECIN / VECOUT",
            "DataCopyPad from LocalTensor to GlobalTensor with DataCopyExtParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
#endif
    }
}

template <typename T>
__aicore__ inline void DataCopyPad(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const DataCopyExtParams &dataCopyParams, const Nd2NzParams &nd2nzParams)
{
    CheckNd2NzParams(nd2nzParams, "DataCopyPad with Nd2NzParams");
    using PrimType = PrimT<T>;
    CheckTensorPos<T>(dstLocal, Hardware::L1, "dstLocal", "TSCM", "DataCopyPad with Nd2NzParams");
    CheckTensorPos<T>(srcLocal, Hardware::UB, "srcLocal", "VECIN / VECOUT", "DataCopyPad with Nd2NzParams");
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyPadTensorSizeOverflow(dstLocal, srcLocal, dataCopyParams, nd2nzParams)));
    DataCopyPadUB2L1Impl((__cbuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        dataCopyParams, nd2nzParams);
}

template <typename T, TPosition pos>
__aicore__ inline void SetPadValue(T paddingValue)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
    set_mov_pad_val(GetScalarBitcodeValue((T)paddingValue));
#elif defined(__DAV_M310__)
    if constexpr(pos == TPosition::MAX || GetPhyType(pos) == Hardware::UB) {
        set_pad_val_outtoub(GetScalarBitcodeValue((T)paddingValue));
    } else if constexpr(GetPhyType(pos) == Hardware::L1) {
        set_pad_val_outtol1(GetScalarBitcodeValue((T)paddingValue));
    } else {
        ASCENDC_REPORT_NOT_SUPPORT(false, "SetPadValue");
    }
#else
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetPadValue");
#endif
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_VCONV_INTERFACE_H