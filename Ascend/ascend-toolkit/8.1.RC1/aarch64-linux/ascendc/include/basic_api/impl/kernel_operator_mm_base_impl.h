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
 * \file kernel_operator_mm_base_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_MM_BASE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_MM_BASE_IMPL_H
#include "kernel_tensor.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_mm_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_mm_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_mm_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_mm_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_mm_impl.h"
#endif
#include "kernel_operator_mm_check.h"
#include "kernel_operator_mm_load2d_impl.h"
#include "kernel_struct_mm.h"
namespace AscendC {
struct IsResetLoad3dConfig {
    __aicore__ constexpr IsResetLoad3dConfig(const bool isSetFMatrixIn, const bool isSetPaddingIn)
    {
        isSetFMatrix = isSetFMatrixIn;
        isSetPadding = isSetPaddingIn;
    }
    bool isSetFMatrix = true;
    bool isSetPadding = true;
};

constexpr IsResetLoad3dConfig IS_RESER_LOAD3D_DEFAULT_CONFIG = {true, true};

/* **************************************************************************************************
 * LoadData 3dv1                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] loadDataParams.padList padding list
 * @param [in] loadDataParams.l1H operand height
 * @param [in] loadDataParams.l1W operand width
 * @param [in] loadDataParams.c1Inde The starting point of the tensor C1 dimension
 * @param [in] loadDataParams.fetchFilterW The starting position of the w dimension on the convolution kernel
 * @param [in] loadDataParams.fetchFilterH The starting position of the H dimension on the convolution kernel
 * @param [in] loadDataParams.leftTopW Start point of the W dimension on the source operand
 * @param [in] loadDataParams.leftTopH Start point of the H dimension on the source operand
 * @param [in] loadDataParams.strideW W dimension stride
 * @param [in] loadDataParams.strideH H dimension stride
 * @param [in] loadDataParams.filterW Convolution kernel width
 * @param [in] loadDataParams.filterH Convolution kernel height
 * @param [in] loadDataParams.dilationFilterW Convolution kernel width expansion coefficient
 * @param [in] loadDataParams.dilationFilterH Convolution kernel height expansion coefficient
 * @param [in] loadDataParams.jumpStride repeat stride
 * @param [in] loadDataParams.repeatMode repeat mode
 * @param [in] loadDataParams.repeatTime repeat times
 * @param [in] loadDataParams.cSize judge whether to turn on optimization
 * @param [in] loadDataParams.padValue Value of Pad filling value
 */

template <typename T, const IsResetLoad3dConfig &defaultConfig = IS_RESER_LOAD3D_DEFAULT_CONFIG,
    typename U = PrimT<T>, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void LoadDataImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData3DParamsV1<U>& loadDataParams)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncLoadData3dv1(dstLocal, srcLocal, loadDataParams, "LoadData with LoadData3DParamsV1")) {
        ASCENDC_REPORT_CHECK_ERROR("LoadData with LoadData3DParamsV1", KernelFuncType::NONE_MODE);
    }
#endif
    ASCENDC_ASSERT((SupportType<PrimT<T>, uint8_t, int8_t, half>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "LoadData with LoadData3DParamsV1, current api support dtype combination is src and dst both: uint8_t / int8_t "
        "/ half.");});

    if constexpr (defaultConfig.isSetFMatrix) {
        Load3DSetFMatrixCal(loadDataParams.l1H, loadDataParams.l1W, loadDataParams.padList);
    }
    if constexpr (defaultConfig.isSetPadding) {
        Load3DSetPaddingCal(loadDataParams.padValue);
    }

    CheckTensorPos<T>(srcLocal, Hardware::L1, "srcLocal", "A1 / B1", "LoadData with LoadData3DParamsV1");
    CheckTensorAlign<T>(srcLocal, ONE_BLK_SIZE, "srcLocal", "LoadData with LoadData3DParamsV1");
    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstScope == Hardware::L0A) {
        CheckTensorAlign<T>(dstLocal, VALUE_512, "dstLocal", "LoadData with LoadData3DParamsV1"); // 512B align
        LoadData3DV1L12L0ACal((__ca__ PrimT<T>*)dstLocal.GetPhyAddr(),
                              (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else if (dstScope == Hardware::L0B) {
        CheckTensorAlign<T>(dstLocal, VALUE_512, "dstLocal", "LoadData with LoadData3DParamsV1"); // 512B align
        LoadData3DV1L12L0BCal((__cb__ PrimT<T>*)dstLocal.GetPhyAddr(),
                              (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else if (dstScope == Hardware::UB) {
        CheckTensorAlign<T>(dstLocal, ONE_BLK_SIZE, "dstLocal", "LoadData with LoadData3DParamsV1");
        LoadData3DV1L12UBCal((__ubuf__ PrimT<T>*)dstLocal.GetPhyAddr(),
                             (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else {
        ASCENDC_CHECK_TPOSITION((false), "dstLocal", "A2 / B2 / UB", "LoadData with LoadData3DParamsV1",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}

/* **************************************************************************************************
 * LoadData 3dv2                                             *
 * enhanced from v1, suitable for aicore > 200                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] loadDataParams.padList padding list
 * @param [in] loadDataParams.l1H operand height
 * @param [in] loadDataParams.l1W operand width
 * @param [in] loadDataParams.channelSize number of channels
 * @param [in] loadDataParams.kExtension Transmission length of K dimension
 * @param [in] loadDataParams.mExtension Transmission length of M dimension
 * @param [in] loadDataParams.kStartPt Start point of K dimension
 * @param [in] loadDataParams.mStartPt Start point of M dimension
 * @param [in] loadDataParams.strideW W dimension stride
 * @param [in] loadDataParams.strideH H dimension stride
 * @param [in] loadDataParams.filterW Convolution kernel width
 * @param [in] loadDataParams.filterH Convolution kernel height
 * @param [in] loadDataParams.dilationFilterW Convolution kernel width expansion coefficient
 * @param [in] loadDataParams.dilationFilterH Convolution kernel height expansion coefficient
 * @param [in] loadDataParams.enTranspose judge whether to enable the transpose function
 * @param [in] loadDataParams.enSmallK Whether to enable the small k feature
 * @param [in] loadDataParams.padValue Value of Pad filling value
 */
template <typename T, const IsResetLoad3dConfig &defaultConfig = IS_RESER_LOAD3D_DEFAULT_CONFIG,
    typename U = PrimT<T>, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void LoadDataImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData3DParamsV2<U>& loadDataParams)
{
    ASCENDC_ASSERT(CheckFuncLoadData3dv2(dstLocal, srcLocal, loadDataParams, "LoadData with LoadData3DParamsV2"), {
        ASCENDC_REPORT_CHECK_ERROR("LoadData with LoadData3DParamsV2", KernelFuncType::NONE_MODE);
    });
    if constexpr (defaultConfig.isSetFMatrix) {
        Load3DSetFMatrixCal(loadDataParams.l1H, loadDataParams.l1W, loadDataParams.padList);
    }
    if constexpr (defaultConfig.isSetPadding) {
        Load3DSetPaddingCal(loadDataParams.padValue);
    }

    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
#if __CCE_AICORE__ == 200
    ASCENDC_ASSERT((SupportType<PrimT<T>, uint8_t, int8_t, half>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "LoadData with LoadData3DParamsV2, current api support dtype combination is src and dst both: uint8_t / int8_t "
        "/ half.");});
#elif __CCE_AICORE__ == 220
    if (dstScope == Hardware::L0A) {
        ASCENDC_ASSERT((SupportType<PrimT<T>, uint8_t, int8_t, half, bfloat16_t, float, uint32_t, int32_t, int4b_t>()),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in LoadData with LoadData3DParamsV2 when dst position is "
            "A2, current api support dtype combination is src and dst both: uint8_t / int8_t / half / bfloat16_t / "
            "float / uint32_t / int32_t / int4b_t.");});
    } else if (dstScope == Hardware::L0B) {
        ASCENDC_ASSERT((SupportType<PrimT<T>, half, bfloat16_t, float, uint32_t, int32_t>()), {KERNEL_LOG(KERNEL_ERROR,
            "Failed to check dtype in LoadData with LoadData3DParamsV2 when dst position is B2, current api support "
            "dtype combination is src and dst both: half / bfloat16_t / float / uint32_t / int32_t.");});
    }
#elif defined(__DAV_M310__)
    if (dstScope == Hardware::L0A) {
        ASCENDC_ASSERT((SupportType<PrimT<T>, uint8_t, int8_t, half, uint16_t, int16_t, int4b_t>()),
            {KERNEL_LOG(KERNEL_ERROR,
            "Failed to check dtype in LoadData with LoadData3DParamsV2 when dst position is A2, current api support "
            "dtype combination is src and dst both: uint8_t / int8_t / half / uint16_t / int16_t / int4b_t.");});
    } else {
        ASCENDC_ASSERT((SupportType<PrimT<T>, half, int16_t, uint16_t>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype "
            "in LoadData with LoadData3DParamsV2 when dst position is B2, current api support dtype combination is src "
            "and dst both: half / int16_t / uint16_t.");});
    }
#endif

    CheckTensorPos<T>(srcLocal, Hardware::L1, "srcLocal", "A1 / B1", "LoadData with LoadData3DParamsV2");
    if (dstScope == Hardware::L0A) {
        CheckTensorAlign<T>(dstLocal, VALUE_512, "dstLocal", "LoadData with LoadData3DParamsV2");
        LoadData3DV2L12L0ACal((__ca__ PrimT<T>*)dstLocal.GetPhyAddr(),
                              (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else if (dstScope == Hardware::L0B) {
        CheckTensorAlign<T>(dstLocal, VALUE_512, "dstLocal", "LoadData with LoadData3DParamsV2");
        LoadData3DV2L12L0BCal((__cb__ PrimT<T>*)dstLocal.GetPhyAddr(),
                              (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else if (dstScope == Hardware::UB) {
        CheckTensorAlign<T>(dstLocal, ONE_BLK_SIZE, "dstLocal", "LoadData with LoadData3DParamsV2");
        LoadData3DV2L12UBCal((__ubuf__ PrimT<T>*)dstLocal.GetPhyAddr(),
                             (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else {
        ASCENDC_CHECK_TPOSITION((false), "dstLocal", "A2 / B2 / UB", "LoadData with LoadData3DParamsV2",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}

#if __CCE_AICORE__ >= 220 && (!defined(__DAV_M310__))
// cce compiler process laod3d bfloat16_t using B8, so use the half dtype instead
template <const IsResetLoad3dConfig& defaultConfig>
[[deprecated("NOTICE: LoadData<IsResetLoad3dConfig> has been deprecated and will be removed in the next version."
             " Please do not use it!")]]
__aicore__ inline void LoadData(const LocalTensor<bfloat16_t>& dstLocal, const LocalTensor<bfloat16_t>& srcLocal,
    const LoadData3DParamsV2<bfloat16_t>& loadDataParams)
{
    LoadDataImpl<bfloat16_t, defaultConfig>(dstLocal, srcLocal, loadDataParams);
}
#endif

/* **************************************************************************************************
 * LoadData 3dv2Pro                                             *
 * enhanced from v1, suitable for aicore > 200                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] loadDataParams.channelSize number of channels
 * @param [in] loadDataParams.GetKExtension() Transmission length of K dimension
 * @param [in] loadDataParams.GetMExtension() Transmission length of M dimension
 * @param [in] loadDataParams.GetKStartPt() Start point of K dimension
 * @param [in] loadDataParams.GetMStartPt() Start point of M dimension
 * @param [in] loadDataParams.GetStrideW() W dimension stride
 * @param [in] loadDataParams.GetStrideH() H dimension stride
 * @param [in] loadDataParams.GetFilterW() Convolution kernel width
 * @param [in] loadDataParams.GetFilterH() Convolution kernel height
 * @param [in] loadDataParams.GetDilationFilterW() Convolution kernel width expansion coefficient
 * @param [in] loadDataParams.GetDilationFilterH() Convolution kernel height expansion coefficient
 * @param [in] loadDataParams.enTranspose judge whether to enable the transpose function
 * @param [in] loadDataParams.enSmallK Whether to enable the small k feature
 */
template <typename T>
__aicore__ inline void LoadDataImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData3DParamsV2Pro& loadDataParams)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncLoadData3dv2Pro(dstLocal, srcLocal, loadDataParams, "LoadData with LoadData3DParamsV2Pro")) {
        ASCENDC_REPORT_CHECK_ERROR("LoadData with LoadData3DParamsV2Pro", KernelFuncType::NONE_MODE);
    }
#endif
    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstScope == Hardware::L0A) {
        LoadData3DV2L12L0ACal((__ca__ PrimT<T>*)dstLocal.GetPhyAddr(),
                              (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else if (dstScope == Hardware::L0B) {
        LoadData3DV2L12L0BCal((__cb__ PrimT<T>*)dstLocal.GetPhyAddr(),
                              (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else if (dstScope == Hardware::UB) {
        LoadData3DV2L12UBCal((__ubuf__ PrimT<T>*)dstLocal.GetPhyAddr(),
                             (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else {
        ASCENDC_CHECK_TPOSITION((false), "dstLocal", "A1 / A2 / UB", "LoadData with LoadData3DParamsV2Pro",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}

/* **************************************************************************************************
 * Mmad                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Mmad
 * @brief Matrix multiplication and addition
 * @param [out] dstLocal output LocalTensor
 * @param [in] fmLocal input LocalTensor
 * @param [in] filterLocal input LocalTensor
 * @param [in] mmadParams.m Left matrix row number
 * @param [in] mmadParams.n right matrix column number
 * @param [in] mmadParams.k Left matrix column number m
 * @param [in] mmadParams.unitFlag whether enable unit flag
 * @param [in] mmadParams.kDirectionAlign is the indicator for alignment in L0A/L0B in the K direction
 * @param [in] mmadParams.cmatrixSource indicates the C matrix source, 1: the C matrix is in bias table buffer, 0: the C
 * matrix is in L0C
 * @param [in] mmadParams.cmatrixInitVal indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real
 * number in C matrix
 */

template <typename DstT, typename Src0T, typename Src1T>
__aicore__ inline void MmadImpl(const LocalTensor<DstT>& dstLocal, const LocalTensor<Src0T>& fmLocal,
    const LocalTensor<Src1T>& filterLocal, const MmadParams& mmadParams)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckMmadParams(dstLocal, fmLocal, filterLocal, mmadParams, "Mmad")) {
        ASCENDC_REPORT_CHECK_ERROR("Mmad", KernelFuncType::NONE_MODE);
    }
    CheckMmadAlign(dstLocal, fmLocal, filterLocal);
#endif
    MmadCal((__cc__ PrimT<DstT>*)dstLocal.GetPhyAddr(), (__ca__ PrimT<Src0T>*)fmLocal.GetPhyAddr(),
        (__cb__ PrimT<Src1T>*)filterLocal.GetPhyAddr(), mmadParams);
}

template <typename DstT, typename Src0T, typename Src1T, typename BiasT>
__aicore__ inline void MmadImpl(const LocalTensor<DstT>& dstLocal, const LocalTensor<Src0T>& fmLocal,
    const LocalTensor<Src1T>& filterLocal, const LocalTensor<BiasT>& biasLocal, const MmadParams& mmadParams)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckMmadParams(dstLocal, fmLocal, filterLocal, biasLocal, mmadParams, "Mmad with biasLocal")) {
        ASCENDC_REPORT_CHECK_ERROR("Mmad with biasLocal", KernelFuncType::NONE_MODE);
    }
    CheckMmadAlign(dstLocal, fmLocal, filterLocal);
    CheckTensorAlign<BiasT>(biasLocal, 128, "biasLocal", "Mmad");
#if __CCE_AICORE__ == 220
    ASCENDC_ASSERT((SupportType<Tuple<PrimT<DstT>, PrimT<Src0T>, PrimT<Src1T>, PrimT<BiasT>>,
        Tuple<int32_t, int8_t, int8_t, int32_t>,
        Tuple<float, half, half, float>, Tuple<float, float, float, float>,
        Tuple<float, bfloat16_t, bfloat16_t, float>>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Mmad, current api support dtype combination is "
        "Dst: int32_t, src0: int8_t, src1: int8_t, Bias: int32_t; Dst: float, src0: half, src1: half, Bias: float; "
        "Dst: float, src0: float, src1: float, Bias: float; "
        "Dst: float, src0: bfloat16_t, src1: bfloat16_t, Bias: float");});
#endif
#endif
    const Hardware biasScope = GetPhyType((TPosition)biasLocal.GetPosition());
    bool cmatrixSource = false;
    if (biasScope == Hardware::BIAS) {
        cmatrixSource = true;
    } else if (biasScope == Hardware::L0C) {
        cmatrixSource = false;
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR,
            "Failed to check biasLocal tensor position in Mmad, supported positions are CO1 or C2"); });
    }
    MmadCal((__cc__ PrimT<DstT>*)dstLocal.GetPhyAddr(), (__ca__ PrimT<Src0T>*)fmLocal.GetPhyAddr(),
        (__cb__ PrimT<Src1T>*)filterLocal.GetPhyAddr(), (uint64_t)biasLocal.GetPhyAddr(), mmadParams, cmatrixSource);
}

#if __CCE_AICORE__ == 220
template <typename T = int32_t, typename U = int8_t,
    typename std::enable_if<IsSameType<PrimT<T>, int32_t>::value, bool>::type = true,
    typename std::enable_if<IsSameType<PrimT<U>, int8_t>::value, bool>::type = true>
__aicore__ inline void MmadSpImpl(const LocalTensor<T>& dstLocal, const LocalTensor<U>& fmLocal,
    const LocalTensor<U>& filterLocal, const MmadParams& mmadParams)
{
    CheckTensorPos<T>(dstLocal, Hardware::L0C, "dstLocal", "CO1", "MmadWithSparse");
    CheckTensorPos<U>(fmLocal, Hardware::L0A, "fmLocal", "A2", "MmadWithSparse");
    CheckTensorPos<U>(filterLocal, Hardware::L0B, "filterLocal", "B2", "MmadWithSparse");
    CheckTensorAlign<T>(dstLocal, 1024, "dstLocal", "MmadWithSparse");             // 1024B aligned
    CheckTensorAlign<U>(fmLocal, VALUE_512, "fmLocal", "MmadWithSparse");           // 512B aligned
    CheckTensorAlign<U>(filterLocal, VALUE_512, "filterLocal", "MmadWithSparse");   // 512B aligned
    ASCENDC_CHECK_VALUE_RANGE(mmadParams.m, 0, UINT12_MAX, "m", "MmadWithSparse");
    ASCENDC_CHECK_VALUE_RANGE(mmadParams.n, 0, UINT12_MAX, "n", "MmadWithSparse");
    ASCENDC_CHECK_VALUE_RANGE(mmadParams.k, 0, UINT12_MAX, "k", "MmadWithSparse");
    MmadSpCal((__cc__ int32_t*)dstLocal.GetPhyAddr(), (__ca__ int8_t*)fmLocal.GetPhyAddr(),
        (__cb__ int8_t*)filterLocal.GetPhyAddr(), mmadParams);
}

template <typename T = int8_t, typename U = uint8_t,
    typename std::enable_if<IsSameType<PrimT<T>, int8_t>::value, bool>::type = true,
    typename std::enable_if<IsSameType<PrimT<U>, uint8_t>::value, bool>::type = true>
__aicore__ inline void LoadDataWithSparseImpl(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<U> &idxLocal, const LoadData2dParams &loadDataParam)
{
    CheckTensorPos<T>(dstLocal, Hardware::L0B, "dstLocal", "B2", "LoadDataWithSparse");
    CheckTensorPos<T>(srcLocal, Hardware::L1, "srcLocal", "B1", "LoadDataWithSparse");
    CheckTensorPos<U>(idxLocal, Hardware::L1, "idxLocal", "B1", "LoadDataWithSparse");
    CheckTensorAlign<T>(dstLocal, VALUE_512, "dstLocal", "LoadDataWithSparse");        // 512B align
    CheckTensorAlign<T>(srcLocal, ONE_BLK_SIZE, "srcLocal", "LoadDataWithSparse");     // 32B align
    CheckTensorAlign<U>(idxLocal, ONE_BLK_SIZE, "idxLocal", "LoadDataWithSparse");    // 32B align
    LoadDataWithSparseCal(dstLocal, srcLocal, idxLocal, loadDataParam);
}
#endif

#if __CCE_AICORE__ == 200
template <typename T = int8_t, typename std::enable_if<IsSameType<PrimT<T>, int8_t>::value, bool>::type = true>
__aicore__ inline void LoadUnzipIndexImpl(const GlobalTensor<T>& srcTensor, uint32_t numOfIndexTabEntry)
{
    LoadUnzipIndexCal(srcTensor, numOfIndexTabEntry);
}
#endif

/* **************************************************************************************************
 * BroadCastVecToMM                                             *
 * ************************************************************************************************* */
template <typename T, typename U>
__aicore__ inline __inout_pipe__(V) void BroadCastVecToMMImpl(const LocalTensor<T> &dstLocal,
    const LocalTensor<U> &srcLocal, const int32_t blockCount, const uint8_t blockLen, const uint8_t srcGap,
    const uint8_t dstGap)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncBroadCastToMM(dstLocal, srcLocal, blockCount, blockLen, srcGap, dstGap, "BroadCastVecToMM")) {
        ASCENDC_REPORT_CHECK_ERROR("BroadCastVecToMM", KernelFuncType::NONE_MODE);
    }
#endif
    BroadCastVecToMMCal((__cc__ PrimT<T>*)dstLocal.GetPhyAddr(), (__ubuf__ PrimT<U>*)srcLocal.GetPhyAddr(),
        blockCount, blockLen, srcGap, dstGap);
}

/* **************************************************************************************************
 * SetLoadDataPaddingValue                                             *                                            *
 * ************************************************************************************************* */
/*
 * @ingroup SetLoadDataPaddingValue
 * @brief setting loadData pad value
 * @param [in]padValue padding value
 */
template <typename T>
__aicore__ inline void Load3DSetPaddingImpl(const T padValue)
{
    Load3DSetPaddingCal(padValue);
}

/* **************************************************************************************************
 * InitConstValue                                             *
 * ************************************************************************************************* */
/*
 * @ingroup InitConstValue
 * @brief L0A/L0B value initializing
 * @param [out] dstLocal output LocalTensor
 * @param [in] InitConstValueParams.repeatTimes repeat times
 * @param [in] InitConstValueParams.repeatTimes blockNum block number
 * @param [in] InitConstValueParams.dstGap interval between the previous tail and the next block head
 * @param [in] InitConstValueParams.initValue initialize Value
 */
template <typename T, typename U = PrimT<T>,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void InitConstValueImpl(const LocalTensor<T> &dstLocal,
    const InitConstValueParams<U> &initConstValueParams)
{
    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstScope == Hardware::L0A) {
        CheckTensorAlign<T>(dstLocal, VALUE_512, "dstLocal", "InitConstValue when TPosition is A2");
        InitL0ANzMatrixCal((__ca__ PrimT<T>*)dstLocal.GetPhyAddr(), initConstValueParams);
    } else if (dstScope == Hardware::L0B) {
        CheckTensorAlign<T>(dstLocal, VALUE_512, "dstLocal", "InitConstValue when TPosition is B2");
        InitL0BNzMatrixCal((__cb__ PrimT<T>*)dstLocal.GetPhyAddr(), initConstValueParams);
    } else if (dstScope == Hardware::L1) {
        CheckTensorAlign<T>(dstLocal, ONE_BLK_SIZE, "dstLocal", "InitConstValue when TPosition is A1 / B1");
        InitL1BufferCal((__cbuf__ PrimT<T>*)dstLocal.GetPhyAddr(), initConstValueParams);
    } else {
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "A1 / B1 / A2 / B2", "InitConstValue",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}

/* **************************************************************************************************
 * SetFmatrix                                             *
 * ************************************************************************************************* */
/*
 * @ingroup SetFmatrix
 * @brief setting fmatrix
 * @param [in]l1H operand height
 * @param [in]l1W operand width
 * @param [in]padList padding list
 * @param [in]fmatrixMode set fmatrix_a or fmatrix_b
 */
__aicore__ inline void SetFmatrixImpl(uint16_t l1H, uint16_t l1W, const uint8_t padList[4],
    const FmatrixMode &fmatrixMode)
{
    if (fmatrixMode == FmatrixMode::FMATRIX_LEFT) {
        Load3DSetFMatrixCal(l1H, l1W, padList);
    } else if (fmatrixMode == FmatrixMode::FMATRIX_RIGHT) {
        Load3DSetFMatrixBCal(l1H, l1W, padList);
    }
}

/* **************************************************************************************************
 * SetLoadDataBoundary                                             *
 * ************************************************************************************************* */
/*
 * @ingroup SetFmatrix
 * @brief setting loaddata boundary
 * @param [in]boundaryValue
 */
__aicore__ inline void SetLoadDataBoundaryImpl(uint32_t boundaryValue)
{
    SetLoadDataBoundaryCal(boundaryValue);
}

/* **************************************************************************************************
 * SetLoadDataRepeat                                             *
 * ************************************************************************************************* */
__aicore__ inline void SetLoadDataRepeatImpl(const LoadDataRepeatParam& repeatParams)
{
    SetLoadDataRepeatCal(repeatParams);
}

/* **************************************************************************************************
 * LoadDataUnzipImpl                                             *
 * ************************************************************************************************* */
/*
 * @ingroup LoadDataUnzip
 * @brief loadData and unzip
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input GlobalTensor
 */
template <typename T>
__aicore__ inline void LoadDataUnzipImpl(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal)
{
    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
#if ASCENDC_CPU_DEBUG
    if (dstScope == Hardware::L1) {
        CheckTensorAlign<T>(dstLocal,  ONE_BLK_SIZE, "dstLocal", "LoadDataUnzip in A1 / B1"); // 32B align
    } else if (dstScope == Hardware::L0A || dstScope == Hardware::L0B) {
        CheckTensorAlign<T>(dstLocal, VALUE_512, "dstLocal", "LoadDataUnzip in B2");               // 512B align
    }
    if constexpr(!SupportType<PrimT<T>, int8_t>()) {
        ASCENDC_ASSERT(false, {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in LoadDataUnzip, current api support "
            "dtype combination is dst: int8_t.");});
    }
#endif
    if (dstScope == Hardware::L1) {
        LoadDataUnzipToL1Cal((__cbuf__ PrimT<T>*)dstLocal.GetPhyAddr(), (__gm__ PrimT<T>*)srcGlobal.GetPhyAddr());
    } else if (dstScope == Hardware::L0A) {
        LoadDataUnzipToL0ACal((__ca__ PrimT<T>*)dstLocal.GetPhyAddr(), (__gm__ PrimT<T>*)srcGlobal.GetPhyAddr());
    } else if (dstScope == Hardware::L0B) {
        LoadDataUnzipToL0BCal((__cb__ PrimT<T>*)dstLocal.GetPhyAddr(), (__gm__ PrimT<T>*)srcGlobal.GetPhyAddr());
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dstLocal tensor position in LoadDataUnzip, "
            "supported positions are A1 / B1 / B2"); });
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_MM_BASE_IMPL_H