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
 * \file kernel_operator_data_copy_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H
#include "kernel_operator_scm_data_copy_impl.h"
#include "kernel_common.h"
#include "kernel_process_lock.h"
namespace AscendC {
// DataCopyParams:    uint16_t blockCount, uint16_t blockLen
// DataCopyExtParams: uint16_t blockCount, uint32_t blockLen
template <typename T>
__aicore__ inline void CheckDataCopyPadParams(uint16_t blockCount, uint32_t blockLen, bool isGMtoUB)
{
#if ASCENDC_CPU_DEBUG
    constexpr uint32_t DATACOPYPAD_BLKLEN_LIMIT = 2097151;    // 21 bits in total, thus range [0, 2097151]
    ASCENDC_CHECK_VALUE_RANGE(blockCount, 0, UINT12_MAX, "blockCount", "DataCopyPad");
    ASCENDC_CHECK_VALUE_RANGE(blockLen, 0, DATACOPYPAD_BLKLEN_LIMIT, "blockLen", "DataCopyPad");
    if (isGMtoUB) {
        ASCENDC_ASSERT((blockLen % sizeof(T) == 0), { KERNEL_LOG(KERNEL_ERROR, "Failed to check blockLen value in "
            "DataCopyPad from GM to VECIN / VECOUT, it must be divisible by %u, current value is %u.", sizeof(T),
            blockLen); });
    }
#endif
}

// DataCopyParams: uint16_t blockCount: 12 bits
__aicore__ inline void CheckDataCopyParams(uint16_t blockCount, uint16_t blockLen)
{
    ASCENDC_CHECK_VALUE_RANGE(blockCount, 1, UINT12_MAX, "blockCount", "DataCopy");
    ASCENDC_CHECK_VALUE_RANGE(blockLen, 1, UINT16_MAX, "blockLen", "DataCopy");
}

__aicore__ inline void ValidateUbL1Address(uint64_t absUbAddr, uint64_t absL1Addr, uint32_t tensorSize)
{
    ASCENDC_ASSERT((absUbAddr < TOTAL_UB_SIZE), {
        KERNEL_LOG(KERNEL_ERROR, "absUbAddr is 0x%lx, which should be in range of [0, %u)", absUbAddr, TOTAL_UB_SIZE);
    });
    ASCENDC_ASSERT(((uint64_t)(absUbAddr + tensorSize) < TOTAL_UB_SIZE), {
        KERNEL_LOG(KERNEL_ERROR, "absUbAddr is 0x%lx, tensorSize is %u, which exceed the limit of ub %d)", absUbAddr,
            tensorSize, TOTAL_UB_SIZE);
    });
    ASCENDC_ASSERT((absL1Addr < TOTAL_L1_SIZE), {
        KERNEL_LOG(KERNEL_ERROR, "absL1Addr is 0x%lx, which should be in range [0, %u)", absL1Addr, TOTAL_L1_SIZE);
    });
    ASCENDC_ASSERT(((uint64_t)(absL1Addr + tensorSize) < TOTAL_L1_SIZE), {
        KERNEL_LOG(KERNEL_ERROR, "absL1Addr is 0x%lx, tensorSize is %u, which exceed the limit of l1 %u)", absL1Addr,
            tensorSize, TOTAL_L1_SIZE);
    });
}

/* **************************************************************************************************
 * DataCopy                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void DataCopyGM2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams)
{
    if ASCEND_IS_AIV {
        CheckDataCopyParams(intriParams.blockCount, intriParams.blockLen);
        ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(dst)) % ONE_BLK_SIZE == 0),
            "dst address should be 32B aligned \n");
        if constexpr (g_gm_overflow_check) {
            __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
            AscendCUtils::CheckGmMemOverflowNormal(src, workSpace, true, false, intriParams);
        }
        copy_gm_to_ubuf((__ubuf__ void*)dst, (__gm__ void*)src, 0, intriParams.blockCount, intriParams.blockLen,
            intriParams.srcStride, intriParams.dstStride);
    }
}

template <typename T>
__aicore__ inline void DataCopyGM2L1Impl(__cbuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams)
{
    if ASCEND_IS_AIC {
        CheckDataCopyParams(intriParams.blockCount, intriParams.blockLen);
        ASCENDC_CHECK_TENSOR_PTR_ALIGN(dst, TPosition::A1, ONE_BLK_SIZE, "dstLocal", "DataCopy from GM to A1 / B1");
        if constexpr (g_gm_overflow_check) {
            __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
            AscendCUtils::CheckGmMemOverflowNormal(src, workSpace, true, false, intriParams);
        }
        copy_gm_to_cbuf((__cbuf__ void*)dst, (__gm__ void*)src, (int8_t)0, (uint16_t)intriParams.blockCount,
            (uint16_t)intriParams.blockLen, (uint16_t)intriParams.srcStride, (uint16_t)intriParams.dstStride, (pad_t)0);
    } else if ASCEND_IS_AIV { // Add for TSCM AIV: just send the message to aic;

#if ASCENDC_CPU_DEBUG
        uint8_t* tscmCpuBaseAddr = GetTPipePtr()->GetBaseAddr(int8_t(TPosition::TSCM));
        uint64_t l1Addr = (uint8_t*)dst - tscmCpuBaseAddr;
        ScmDataCopyMsg((__cbuf__ void*)l1Addr, (__gm__ void*)src, intriParams, -1);
#else
        // scm data copy will call kfc msg, must have kfc server
        // this macro is used to infer pure cube channel op
#ifndef __NO_KFC_SERVER__
        ScmDataCopyMsg((__cbuf__ void*)dst, (__gm__ void*)src, intriParams, -1);
#endif
#endif
    }
}

template <typename T>
__aicore__ inline void DataCopyUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((dst != nullptr), { KERNEL_LOG(KERNEL_ERROR, "dst ptr can not be nullptr"); });
        ASCENDC_ASSERT((src != nullptr), { KERNEL_LOG(KERNEL_ERROR, "src ptr can not be nullptr"); });
        CheckDataCopyParams(intriParams.blockCount, intriParams.blockLen);
        ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(src)) % ONE_BLK_SIZE == 0),
            "src address should be 32B aligned \n");

        if constexpr (g_gm_overflow_check) {
            __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
            AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, false, intriParams);
        }
        copy_ubuf_to_gm((__gm__ void*)dst, (__ubuf__ void*)src, 0, intriParams.blockCount, intriParams.blockLen,
            intriParams.srcStride, intriParams.dstStride);
    }
}

template <typename T>
__aicore__ inline void DataCopyUB2UBImpl(__ubuf__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    if ASCEND_IS_AIV {
        CheckDataCopyParams(intriParams.blockCount, intriParams.blockLen);
        ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(src)) % ONE_BLK_SIZE == 0),
            "src address should be 32B aligned \n");
        ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(dst)) % ONE_BLK_SIZE == 0),
            "dst address should be 32B aligned \n");
        copy_ubuf_to_ubuf((__ubuf__ void*)dst, (__ubuf__ void*)src, 0, intriParams.blockCount, intriParams.blockLen,
            intriParams.srcStride, intriParams.dstStride);
    }
}

template <typename T>
__aicore__ inline void DataCopyUB2L1Impl(__cbuf__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    // Add for TSCM AIV: Copy buffer into workspace, and send message to AIC
    ASCENDC_ASSERT((GetTPipePtr() != nullptr), { KERNEL_LOG(KERNEL_ERROR, "tpipe ptr can not be nullptr"); });
    ASCENDC_ASSERT((dst != nullptr), { KERNEL_LOG(KERNEL_ERROR, "dst ptr can not be nullptr"); });
    ASCENDC_ASSERT((src != nullptr), { KERNEL_LOG(KERNEL_ERROR, "src ptr can not be nullptr"); });
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((GetKfcClient() != nullptr), { KERNEL_LOG(KERNEL_ERROR, "kfc client ptr can not be nullptr"); });
        ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(src)) % ONE_BLK_SIZE == 0),
            "src address should be 32B aligned \n");
        // 1.获取GM的地址
        uint32_t tensorSize = intriParams.blockCount * intriParams.blockLen * 32;
        int32_t ubAddr = -1;
#if ASCENDC_CPU_DEBUG
        uint64_t absSrc = (uint8_t*)src - (uint8_t*)(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::VECIN));
        ASCENDC_ASSERT(((uint64_t)absSrc >= 0 && (uint64_t)absSrc < TOTAL_UB_SIZE), {
            KERNEL_LOG(KERNEL_ERROR, "abs src is 0x%lx, which should be in range of [0, %u)", absSrc, TOTAL_UB_SIZE);
        });
        GM_ADDR gmAddr = (GetKfcClient()->ubStart) + (uint64_t)absSrc;
#else
        GM_ADDR gmAddr = (GetKfcClient()->AllocUB(tensorSize, ubAddr));
#endif
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        // 2.进行对等拷贝ub->GM
        CheckDataCopyParams(intriParams.blockCount, intriParams.blockLen);
        copy_ubuf_to_gm((__gm__ void*)gmAddr, (__ubuf__ void*)src, 0, intriParams.blockCount, intriParams.blockLen,
            intriParams.srcStride, intriParams.srcStride);

        // 3.发送消息,从GM拷贝至L1
#if ASCENDC_CPU_DEBUG
        uint8_t* aicBaseTSCMAddr = GetTPipePtr()->GetBaseAddr(int8_t(TPosition::TSCM));
        ASCENDC_ASSERT((aicBaseTSCMAddr != nullptr),
            { KERNEL_LOG(KERNEL_ERROR, "aicBaseTSCMAddr can not be nullptr"); });
        uint64_t l1AddrDst = (uint8_t*)dst - (uint8_t*)aicBaseTSCMAddr;
        ASCENDC_ASSERT((l1AddrDst < TOTAL_L1_SIZE), { KERNEL_LOG(KERNEL_ERROR,
            "l1AddrDst is 0x%lx, which should be in range of [0, %u)", l1AddrDst, TOTAL_L1_SIZE); });
        ScmDataCopyMsg((__cbuf__ void*)l1AddrDst, (__gm__ void*)gmAddr, intriParams, ubAddr);
#else
        ScmDataCopyMsg((__cbuf__ void*)dst, (__gm__ void*)gmAddr, intriParams, ubAddr);
#endif
    }
}

template <typename T>
__aicore__ inline void DataCopyUB2L1ND2NZImpl(__cbuf__ T* dst, __ubuf__ T* src, const Nd2NzParams& intriParams)
{
    // Add for TSCM AIV: Copy buffer into workspace, and send message to AIC
    ASCENDC_ASSERT((GetTPipePtr() != nullptr), { KERNEL_LOG(KERNEL_ERROR, "tpipe ptr can not be nullptr"); });
    ASCENDC_ASSERT((dst != nullptr), { KERNEL_LOG(KERNEL_ERROR, "dst ptr can not be nullptr"); });
    ASCENDC_ASSERT((src != nullptr), { KERNEL_LOG(KERNEL_ERROR, "src ptr can not be nullptr"); });
    if ASCEND_IS_AIV {
        ASSERT(GetKfcClient() != nullptr);
        ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(src)) % ONE_BLK_SIZE == 0),
            "src address should be 32B aligned \n");
        uint32_t tensorSize = intriParams.nValue * intriParams.dValue;
        int32_t ubAddr = -1;
        // 1.获取GM的对应地址,将相应的数据搬运到GM, 采用ND->ND的方式
#if ASCENDC_CPU_DEBUG
        uint64_t absUbAddr = (uint8_t*)src - (uint8_t*)(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::VECIN));
        uint64_t absL1Addr = (uint8_t*)dst - (uint8_t*)(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::TSCM));
        ValidateUbL1Address(absUbAddr, absL1Addr, tensorSize);
        ASCENDC_ASSERT((absL1Addr % ONE_BLK_SIZE == 0), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dst tensor address "
            "alignment in DataCopy with Nd2NzParams from VECIN / VECOUT to TSCM, it should be 32B aligned");});
        ASCENDC_ASSERT((intriParams.ndNum == 1), {KERNEL_LOG(KERNEL_ERROR,
            "intriParams.ndNum is %hu, which can only be 1", intriParams.ndNum);});
        GM_ADDR gmAddr = (GetKfcClient()->AllocUB(tensorSize, ubAddr));
#else
        GM_ADDR gmAddr = (GetKfcClient()->AllocUB(tensorSize, ubAddr));
#endif
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        // 2.进行对等拷贝ub->GM
        copy_ubuf_to_gm((__gm__ void*)gmAddr, (__ubuf__ void*)src, 0, 1, tensorSize * sizeof(T) / 32, 0, 0);

        // 3.发送消息,从GM拷贝至L1
#if ASCENDC_CPU_DEBUG
        ScmDataCopyND2NZMsg((__cbuf__ void*)absL1Addr, (__gm__ void*)gmAddr, sizeof(T), intriParams, ubAddr);
#else
        ScmDataCopyND2NZMsg((__cbuf__ void*)dst, (__gm__ void*)gmAddr, sizeof(T), intriParams, ubAddr);
#endif
    }
}

template <typename T>
__aicore__ inline void DataCopyL12UBImpl(__ubuf__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "DataCopy from A1 / B1 to VECIN / VECOUT");
}

template <typename T>
__aicore__ inline void DataCopyL12BTImpl(const uint64_t dst, __cbuf__ T* src, const uint16_t isenableConv,
    const DataCopyParams &intriParams)
{
    if ASCEND_IS_AIC {
        CheckDataCopyParams(intriParams.blockCount, intriParams.blockLen);
        copy_cbuf_to_bt(dst, (__cbuf__ void*)src, isenableConv, intriParams.blockCount, intriParams.blockLen,
            intriParams.srcStride, intriParams.dstStride);
    }
}

template <typename T>
__aicore__ inline void DataCopyL12FBImpl(__fbuf__ T* dst, __cbuf__ T* src, const DataCopyParams &intriParams)
{
    if ASCEND_IS_AIC {
        CheckDataCopyParams(intriParams.blockCount, intriParams.blockLen);
        copy_cbuf_to_fbuf((__fbuf__ void*)dst, (__cbuf__ void*)src, intriParams.blockCount, intriParams.blockLen,
            intriParams.srcStride, intriParams.dstStride);
    }
}

template <typename T>
__aicore__ inline void DataCopyGM2L1ND2NZImplBase(__cbuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    if ASCEND_IS_AIC {
        if constexpr (g_gm_overflow_check) {
            __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
            AscendCUtils::CheckGmMemOverflowNd2Nz(src, workSpace, true, intriParams);
        }
        if constexpr (sizeof(T) == B8_BYTE_SIZE) {
            copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ T*)dst, (__gm__ T*)src, 0, intriParams.ndNum, intriParams.nValue,
                intriParams.dValue, intriParams.srcNdMatrixStride, intriParams.srcDValue, intriParams.dstNzC0Stride,
                intriParams.dstNzNStride, intriParams.dstNzMatrixStride);
        } else if constexpr (sizeof(T) == B16_BYTE_SIZE) {
            copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ T*)dst, (__gm__ T*)src, 0, intriParams.ndNum, intriParams.nValue,
                intriParams.dValue, intriParams.srcNdMatrixStride, intriParams.srcDValue, intriParams.dstNzC0Stride,
                intriParams.dstNzNStride, intriParams.dstNzMatrixStride);
        } else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
            copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ T*)dst, (__gm__ T*)src, 0, intriParams.ndNum, intriParams.nValue,
                intriParams.dValue, intriParams.srcNdMatrixStride, intriParams.srcDValue, intriParams.dstNzC0Stride,
                intriParams.dstNzNStride, intriParams.dstNzMatrixStride);
        }
    } else if ASCEND_IS_AIV { // Add for TSCM: aiv just send the message
#if ASCENDC_CPU_DEBUG
        uint8_t* tscmCpuBaseAddr = GetTPipePtr()->GetBaseAddr(int8_t(TPosition::TSCM));
        uint64_t l1Addr = (uint8_t*)dst - tscmCpuBaseAddr;
        ScmDataCopyND2NZMsg((__cbuf__ void*)l1Addr, (__gm__ void*)src, sizeof(T), intriParams, -1);
#else
        ScmDataCopyND2NZMsg(dst, src, sizeof(T), intriParams, -1);
#endif
    }
}

template <typename T>
__aicore__ inline void DataCopyGM2L1ND2NZImpl(__cbuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    ASCENDC_CHECK_TENSOR_PTR_ALIGN(dst, TPosition::A1, ONE_BLK_SIZE, "dstLocal",
        "DataCopy from GM to A1 / B1 with Nd2NzParams");
    if constexpr (SupportType<T, int4b_t>()) {
        DataCopyGM2L1ND2NZImplBase((__cbuf__ int8_t *)dst, (__gm__ int8_t *)src, intriParams);
    } else if (sizeof(T) == B8_BYTE_SIZE || sizeof(T) == B16_BYTE_SIZE || sizeof(T) == B32_BYTE_SIZE){
        DataCopyGM2L1ND2NZImplBase(dst, src, intriParams);
    } else {
        ASCENDC_ASSERT(false, {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in DataCopy with Nd2NzParams, current "
            "api support dtype combination is src and dst both: int4b_t / int8_t / uint8_t / int16_t / uint16_t / "
            "half / bfloat16_t / int32_t / uint32_t / float");});
    }
}

template <typename T>
__aicore__ inline void DataCopyL12GMImpl(__gm__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    if ASCEND_IS_AIC {
        CheckDataCopyParams(intriParams.blockCount, intriParams.blockLen);
        ASCENDC_CHECK_TENSOR_PTR_ALIGN(src, TPosition::A1, ONE_BLK_SIZE, "srcLocal", "DataCopy from A1 / B1 to GM");
        if constexpr (g_gm_overflow_check) {
            __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
            AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, false, intriParams);
        }
        copy_cbuf_to_gm((__gm__ void*)dst, (__cbuf__ void*)src, 0, intriParams.blockCount, intriParams.blockLen,
            intriParams.srcStride, intriParams.dstStride);
    }
}

/* **************************************************************************************************
 * Copy                                             *
 * ************************************************************************************************* */

template <typename T>
__aicore__ inline void CopyIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t repeatTimes,
    const CopyRepeatParams& repeatParams)
{
    ASCENDC_CHECK_VALUE_RANGE(repeatParams.srcRepeatSize, 0, UINT12_MAX, "srcRepeatSize", "Copy");
    ASCENDC_CHECK_VALUE_RANGE(repeatParams.dstRepeatSize, 0, UINT12_MAX, "dstRepeatSize", "Copy");
    if constexpr(sizeof(T) == B16_BYTE_SIZE) {
        vcopy((__ubuf__ uint16_t*)dst, (__ubuf__ uint16_t*)src, repeatTimes, repeatParams.dstStride,
            repeatParams.srcStride, repeatParams.dstRepeatSize, repeatParams.srcRepeatSize);
    } else if constexpr(sizeof(T) == B32_BYTE_SIZE) {
        vcopy((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src, repeatTimes, repeatParams.dstStride,
            repeatParams.srcStride, repeatParams.dstRepeatSize, repeatParams.srcRepeatSize);
    } else {
        ASCENDC_ASSERT(false, {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Copy, current api support dtype "
            "combination is src and dst both: int16_t / uint16_t / half / bfloat16_t / int32_t / uint32_t / float");});
    }
}

// Copy::Level 0 - mask bit mode
template <typename T, bool isSetMask = true>
__aicore__ inline void CopyImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[], uint8_t repeatTimes,
    const CopyRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        CopyIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
    }
}

// Copy::Level 0 - mask count mode
template <typename T, bool isSetMask = true>
__aicore__ inline void CopyImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask, uint8_t repeatTimes,
    const CopyRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        CopyIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
    }
}

// -------- l1 to l0c --------
// ---- matrix ----
template <typename DstT, typename SrcT>
__aicore__ inline void DataCopyL12L0CImpl(__cc__ DstT* dst, __cbuf__ SrcT* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    if ASCEND_IS_AIC {
        static_assert((SupportType<Tuple<SrcT, DstT>, Tuple<half, half>, Tuple<float, half>, Tuple<float, bfloat16_t>,
            Tuple<float, float>, Tuple<bfloat16_t, bfloat16_t>, Tuple<int32_t, int32_t>, Tuple<uint32_t, uint32_t>,
            Tuple<uint32_t, uint32_t>>()), "Failed to check dtype in DataCopy from A1 / B1 to CO1, current api support "
            "dtype combination is src: half, dst: half; src: float, dst: half / bfloat16_t / float; src: bfloat16_t, "
            "dst: bfloat16_t; src: int32_t, dst: int32_t; src: uint32_t, dst: uint32_t.");
        ASCENDC_CHECK_TENSOR_PTR_ALIGN(src, TPosition::A1, ONE_BLK_SIZE, "srcLocal", "DataCopy from A1 / B1 to CO1");
        ASCENDC_CHECK_TENSOR_PTR_ALIGN(dst, TPosition::CO1, 1024, "dstLocal", "DataCopy from A1 / B1 to CO1");
        copy_matrix_cbuf_to_cc((__cc__ DstT*)dst, (__cbuf__ SrcT*)src, intriParams.blockCount, intriParams.blockLen,
            intriParams.srcStride, intriParams.dstStride);
    }
}

/* **************************************************************************************************
 * DataCopy                                             *
 * ************************************************************************************************* */
template <typename T, typename U>
__aicore__ inline void DataCopyL0C2UBImpl(__ubuf__ T* dst, __cc__ U* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "DataCopy from CO1 to CO2");
}

template <typename T, typename U>
__aicore__ inline void DataCopyUB2L0CImpl(__cc__ T* dst, __ubuf__ U* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "DataCopy from CO2 to CO1");
}

template <typename T>
__aicore__ inline void DataCopySliceGm2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParamsIn)
{
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(dst)) % ONE_BLK_SIZE == 0),
        "dst address should be 32B aligned \n");
    DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };
    uint16_t burstLen = intriParamsIn.blockLen * ONE_BLK_SIZE;
    DataCopyExtParams intriParams{ intriParamsIn.blockCount, burstLen, intriParamsIn.srcStride, intriParamsIn.dstStride,
        0 };
    DataCopyPadGm2UBImpl(dst, src, intriParams, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPadGm2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams,
    const DataCopyPadParams& padParams)
{
    if ASCEND_IS_AIC {
        return;
    }
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(dst)) % ONE_BLK_SIZE == 0), "Failed "
        "to check dst tensor address alignment in DataCopyPad from GM to VECIN / VECOUT, it should be 32B aligned.\n");
    CheckDataCopyPadParams<T>(intriParams.blockCount, static_cast<uint32_t>(intriParams.blockLen), true);
    if (padParams.isPad) {
        set_mov_pad_val(padParams.paddingValue);
    }
    if constexpr (g_gm_overflow_check && (sizeof(T) == B8_BYTE_SIZE || sizeof(T) == B16_BYTE_SIZE
        || sizeof(T) == B32_BYTE_SIZE || sizeof(T) == B64_BYTE_SIZE)) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(src, workSpace, true, true, intriParams);
    }
    if constexpr (sizeof(T) == B8_BYTE_SIZE) {
        copy_gm_to_ubuf_align_b8(dst, src, 0, intriParams.blockCount, intriParams.blockLen, padParams.leftPadding,
            padParams.rightPadding, intriParams.srcStride, intriParams.dstStride);
    } else if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        copy_gm_to_ubuf_align_b16(dst, src, 0, intriParams.blockCount, intriParams.blockLen, padParams.leftPadding,
            padParams.rightPadding, intriParams.srcStride, intriParams.dstStride);
    } else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, intriParams.blockCount, intriParams.blockLen, padParams.leftPadding,
            padParams.rightPadding, intriParams.srcStride, intriParams.dstStride);
    } else if constexpr (sizeof(T) == B64_BYTE_SIZE) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, intriParams.blockCount, intriParams.blockLen,
            (padParams.leftPadding << 1), (padParams.rightPadding << 1), intriParams.srcStride, intriParams.dstStride);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in DataCopyPad from GM to VECIN / "
            "VECOUT, current api support dtype combination is src and dst both: int8_t / uint8_t / half / bfloat16_t / "
            "int16_t / uint16_t / float / int32_t / uint32_t / double / int64_t / uint64_t.");});
    }
}

template <typename T>
__aicore__ inline void DataCopyPadGm2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyExtParams& intriParams,
    const DataCopyPadExtParams<T>& padParams)
{
    if ASCEND_IS_AIC {
        return;
    }
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(dst)) % ONE_BLK_SIZE == 0), "Failed "
        "to check dst tensor address alignment in DataCopyPad from GM to VECIN / VECOUT, it should be 32B aligned.\n");
    CheckDataCopyPadParams<T>(intriParams.blockCount, intriParams.blockLen, true);
    if (padParams.isPad) {
        set_mov_pad_val(GetScalarBitcodeValue(static_cast<T>(padParams.paddingValue)));
    }
    if constexpr (g_gm_overflow_check && (sizeof(T) == B8_BYTE_SIZE || sizeof(T) == B16_BYTE_SIZE
        || sizeof(T) == B32_BYTE_SIZE || sizeof(T) == B64_BYTE_SIZE)) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(src, workSpace, true, true, intriParams);
    }
    if constexpr (sizeof(T) == B8_BYTE_SIZE) {
        copy_gm_to_ubuf_align_b8(dst, src, 0, intriParams.blockCount, intriParams.blockLen, padParams.leftPadding,
            padParams.rightPadding, intriParams.srcStride, intriParams.dstStride);
    } else if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        copy_gm_to_ubuf_align_b16(dst, src, 0, intriParams.blockCount, intriParams.blockLen, padParams.leftPadding,
            padParams.rightPadding, intriParams.srcStride, intriParams.dstStride);
    } else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, intriParams.blockCount, intriParams.blockLen, padParams.leftPadding,
            padParams.rightPadding, intriParams.srcStride, intriParams.dstStride);
    } else if constexpr (sizeof(T) == B64_BYTE_SIZE) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, intriParams.blockCount, intriParams.blockLen,
            (padParams.leftPadding << 1), (padParams.rightPadding << 1), intriParams.srcStride, intriParams.dstStride);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in DataCopyPad from GM to VECIN / "
            "VECOUT, current api support dtype combination is src and dst both: int8_t / uint8_t / half / bfloat16_t / "
            "int16_t / uint16_t / float / int32_t / uint32_t / double / int64_t / uint64_t.");});
    }
}

template <typename T>
__aicore__ inline void DataCopySliceUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParamsIn)
{
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(src)) % ONE_BLK_SIZE == 0),
        "src address should be 32B aligned \n");
    uint32_t burstLen = intriParamsIn.blockLen * ONE_BLK_SIZE;
    DataCopyExtParams intriParams{ intriParamsIn.blockCount, burstLen, intriParamsIn.srcStride, intriParamsIn.dstStride,
        0 };
    DataCopyPadUB2GMImpl(dst, src, intriParams);
}

template <typename T>
__aicore__ inline void DataCopyPadUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    if ASCEND_IS_AIC {
        return;
    }
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(src)) % ONE_BLK_SIZE == 0), "Failed "
        "to check src tensor address alignment in DataCopyPad from VECIN / VECOUT to GM, it should be 32B aligned.\n");
    CheckDataCopyPadParams<T>(intriParams.blockCount, static_cast<uint32_t>(intriParams.blockLen), false);
    if constexpr (g_gm_overflow_check && (sizeof(T) == B8_BYTE_SIZE || sizeof(T) == B16_BYTE_SIZE
        || sizeof(T) == B32_BYTE_SIZE || sizeof(T) == B64_BYTE_SIZE)) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, true, intriParams);
    }
    if constexpr (sizeof(T) == B8_BYTE_SIZE) {
        copy_ubuf_to_gm_align_b8(dst, src, 0, intriParams.blockCount, intriParams.blockLen, (uint8_t)0, (uint8_t)0,
            (uint32_t)intriParams.srcStride, (uint32_t)intriParams.dstStride);
    } else if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        copy_ubuf_to_gm_align_b16(dst, src, 0, intriParams.blockCount, intriParams.blockLen, (uint8_t)0, (uint8_t)0,
            (uint32_t)intriParams.srcStride, (uint32_t)intriParams.dstStride);
    } else if constexpr (sizeof(T) == B32_BYTE_SIZE || sizeof(T) == B64_BYTE_SIZE) {
        copy_ubuf_to_gm_align_b32(dst, src, 0, intriParams.blockCount, intriParams.blockLen, (uint8_t)0, (uint8_t)0,
            (uint32_t)intriParams.srcStride, (uint32_t)intriParams.dstStride);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in DataCopyPad from VECIN / VECOUT to"
            " GM, current api support dtype combination is src and dst both: int8_t / uint8_t / half / bfloat16_t / "
            "int16_t / uint16_t / float / int32_t / uint32_t / double / int64_t / uint64_t.");});
    }
}

template <typename T>
__aicore__ inline void DataCopyPadUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyExtParams& intriParams)
{
    if ASCEND_IS_AIC {
        return;
    }
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(src)) % ONE_BLK_SIZE == 0), "Failed "
        "to check src tensor address alignment in DataCopyPad from VECIN / VECOUT to GM, it should be 32B aligned.\n");
    CheckDataCopyPadParams<T>(intriParams.blockCount, intriParams.blockLen, false);
    if constexpr (g_gm_overflow_check && (sizeof(T) == B8_BYTE_SIZE || sizeof(T) == B16_BYTE_SIZE
        || sizeof(T) == B32_BYTE_SIZE || sizeof(T) == B64_BYTE_SIZE)) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, true, intriParams);
    }
    if constexpr (sizeof(T) == B8_BYTE_SIZE) {
        copy_ubuf_to_gm_align_b8(dst, src, 0, intriParams.blockCount, intriParams.blockLen, (uint8_t)0, (uint8_t)0,
            (uint32_t)intriParams.srcStride, (uint32_t)intriParams.dstStride);
    } else if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        copy_ubuf_to_gm_align_b16(dst, src, 0, intriParams.blockCount, intriParams.blockLen, (uint8_t)0, (uint8_t)0,
            (uint32_t)intriParams.srcStride, (uint32_t)intriParams.dstStride);
    } else if constexpr (sizeof(T) == B32_BYTE_SIZE || sizeof(T) == B64_BYTE_SIZE) {
        copy_ubuf_to_gm_align_b32(dst, src, 0, intriParams.blockCount, intriParams.blockLen, (uint8_t)0, (uint8_t)0,
            intriParams.srcStride, intriParams.dstStride);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in DataCopyPad from VECIN / VECOUT to"
            " GM, current api support dtype combination is src and dst both: int8_t / uint8_t / half / bfloat16_t / "
            "int16_t / uint16_t / float / int32_t / uint32_t / double / int64_t / uint64_t.");});
    }
}

template <typename T>
__aicore__ inline void DataCopyPadUB2L1Impl(__cbuf__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams,
    const Nd2NzParams& nd2nzParams)
{
    // Add for TSCM AIV: Copy buffer into workspace, and send message to AIC
    ASCENDC_ASSERT((GetTPipePtr() != nullptr), { KERNEL_LOG(KERNEL_ERROR, "tpipe ptr can not be nullptr"); });
    ASCENDC_ASSERT((dst != nullptr), { KERNEL_LOG(KERNEL_ERROR, "dst ptr can not be nullptr"); });
    ASCENDC_ASSERT((src != nullptr), { KERNEL_LOG(KERNEL_ERROR, "src ptr can not be nullptr"); });
    CheckDataCopyPadParams<T>(intriParams.blockCount, static_cast<uint32_t>(intriParams.blockLen), false);
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((GetKfcClient() != nullptr), { KERNEL_LOG(KERNEL_ERROR, "kfc client ptr can not be nullptr"); });
        ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(src)) % ONE_BLK_SIZE == 0),
            "Failed to check src tensor address alignment in DataCopyPad from VECIN / VECOUT to TSCM, it should be 32B "
            "aligned.\n");
        uint32_t tensorSize = nd2nzParams.nValue * nd2nzParams.dValue;
        int32_t ubAddr = -1;
        // 1.获取GM的对应地址,将相应的数据搬运到GM, 采用ND->ND的方式
#if ASCENDC_CPU_DEBUG
        uint64_t absUbAddr = (uint8_t*)src - (uint8_t*)(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::VECIN));
        uint64_t absL1Addr = (uint8_t*)dst - (uint8_t*)(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::TSCM));
        ValidateUbL1Address(absUbAddr, absL1Addr, tensorSize);
        ASCENDC_ASSERT((absL1Addr % ONE_BLK_SIZE == 0), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dst tensor address "
            "alignment in DataCopyPad from VECIN / VECOUT to TSCM, it should be 32B aligned");});
        ASCENDC_ASSERT((nd2nzParams.ndNum == 1), {
            KERNEL_LOG(KERNEL_ERROR, "nd2nzParams.ndNum is %hu, which can only be 1", nd2nzParams.ndNum);
        });
        GM_ADDR gmAddr = (GetKfcClient()->AllocUB(tensorSize, ubAddr));
#else
        GM_ADDR gmAddr = (GetKfcClient()->AllocUB(tensorSize, ubAddr));
#endif
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        // 2.进行对等拷贝ub->GM
        ASCENDC_ASSERT((intriParams.blockCount != 0),
                       { KERNEL_LOG(KERNEL_ERROR, "intriParams.blockCount can not be 0"); });
        ASCENDC_ASSERT((intriParams.blockLen != 0), { KERNEL_LOG(KERNEL_ERROR, "intriParams.blockLen can not be 0"); });
        DataCopyPadUB2GMImpl((__gm__ T*)gmAddr, (__ubuf__ T*)src, intriParams);

#if ASCENDC_CPU_DEBUG
        ScmDataCopyND2NZMsg((__cbuf__ void*)absL1Addr, (__gm__ void*)gmAddr, sizeof(T), nd2nzParams, ubAddr);
#else
        ScmDataCopyND2NZMsg((__cbuf__ void*)dst, (__gm__ void*)gmAddr, sizeof(T), nd2nzParams, ubAddr);
#endif
    }
}

template <typename T>
__aicore__ inline void DataCopyPadUB2L1Impl(__cbuf__ T* dst, __ubuf__ T* src, const DataCopyExtParams& intriParams,
    const Nd2NzParams& nd2nzParams)
{
    // Add for TSCM AIV: Copy buffer into workspace, and send message to AIC
    ASCENDC_ASSERT((GetTPipePtr() != nullptr), { KERNEL_LOG(KERNEL_ERROR, "tpipe ptr can not be nullptr"); });
    ASCENDC_ASSERT((dst != nullptr), { KERNEL_LOG(KERNEL_ERROR, "dst ptr can not be nullptr"); });
    ASCENDC_ASSERT((src != nullptr), { KERNEL_LOG(KERNEL_ERROR, "src ptr can not be nullptr"); });
    CheckDataCopyPadParams<T>(intriParams.blockCount, intriParams.blockLen, false);
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((GetKfcClient() != nullptr), { KERNEL_LOG(KERNEL_ERROR, "kfc client ptr can not be nullptr"); });
        ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(src)) % ONE_BLK_SIZE == 0),
            "Failed to check src tensor address alignment in DataCopyPad from VECIN / VECOUT to TSCM, it should be 32B "
            "aligned.\n");
        uint32_t tensorSize = nd2nzParams.nValue * nd2nzParams.dValue;
        int32_t ubAddr = -1;
        // 1.获取GM的对应地址,将相应的数据搬运到GM, 采用ND->ND的方式
#if ASCENDC_CPU_DEBUG
        uint64_t absUbAddr = (uint8_t*)src - (uint8_t*)(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::VECIN));
        uint64_t absL1Addr = (uint8_t*)dst - (uint8_t*)(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::TSCM));
        ValidateUbL1Address(absUbAddr, absL1Addr, tensorSize);
        ASCENDC_ASSERT((absL1Addr % ONE_BLK_SIZE == 0), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dst tensor address "
            "alignment in DataCopyPad from VECIN / VECOUT to TSCM, it should be 32B aligned");});
        ASCENDC_ASSERT((nd2nzParams.ndNum == 1), {
            KERNEL_LOG(KERNEL_ERROR, "nd2nzParams.ndNum is %hu, which can only be 1", nd2nzParams.ndNum);
        });
        GM_ADDR gmAddr = (GetKfcClient()->AllocUB(tensorSize, ubAddr));
#else
        GM_ADDR gmAddr = (GetKfcClient()->AllocUB(tensorSize, ubAddr));
#endif
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        // 2.进行对等拷贝ub->GM
        ASCENDC_ASSERT((intriParams.blockCount != 0),
                       { KERNEL_LOG(KERNEL_ERROR, "intriParams.blockCount can not be 0"); });
        ASCENDC_ASSERT((intriParams.blockLen != 0), { KERNEL_LOG(KERNEL_ERROR, "intriParams.blockLen can not be 0"); });
        DataCopyPadUB2GMImpl((__gm__ T*)gmAddr, (__ubuf__ T*)src, intriParams);

#if ASCENDC_CPU_DEBUG
        ScmDataCopyND2NZMsg((__cbuf__ void*)absL1Addr, (__gm__ void*)gmAddr, sizeof(T), nd2nzParams, ubAddr);
#else
        ScmDataCopyND2NZMsg((__cbuf__ void*)dst, (__gm__ void*)gmAddr, sizeof(T), nd2nzParams, ubAddr);
#endif
    }
}
/**
    AIC
* */
__aicore__ inline void ScmDataCopy(__gm__ void* kfcMsgPtr)
{
    ASCENDC_ASSERT((g_coreType == AIC), { KERNEL_LOG(KERNEL_ERROR, "core type must be AIC"); });
    auto scmCopyParams = reinterpret_cast<__gm__ struct Gm2L1Params*>(kfcMsgPtr);
    ASCENDC_ASSERT((reinterpret_cast<uint64_t>(scmCopyParams->dst) < TOTAL_L1_SIZE), {
        KERNEL_LOG(KERNEL_ERROR, "dst ptr is %p, which should be less than total l1 size %u", scmCopyParams->dst,
            TOTAL_L1_SIZE);
    });
    auto dst = reinterpret_cast<__cbuf__ void*>(scmCopyParams->dst);
    auto& intriParams = scmCopyParams->intri;
#if ASCENDC_CPU_DEBUG
    uint8_t* aicBaseTSCMAddr = GetTPipePtr()->GetBaseAddr(int8_t(TPosition::TSCM));
    uint8_t* l1AddrDst = aicBaseTSCMAddr + (uint64_t)scmCopyParams->dst;
    copy_gm_to_cbuf((__cbuf__ void*)l1AddrDst, (__gm__ void*)scmCopyParams->src, (int8_t)0,
        (uint16_t)intriParams.blockCount, (uint16_t)intriParams.blockLen, (uint16_t)intriParams.srcStride,
        (uint16_t)intriParams.dstStride, (pad_t)0);
#else
    copy_gm_to_cbuf((__cbuf__ void*)dst, (__gm__ void*)scmCopyParams->src, (int8_t)0, (uint16_t)intriParams.blockCount,
        (uint16_t)intriParams.blockLen, (uint16_t)intriParams.srcStride, (uint16_t)intriParams.dstStride, (pad_t)0);
#endif
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
    SetFlag<HardEvent::MTE2_MTE1>(eventID);
    WaitFlag<HardEvent::MTE2_MTE1>(eventID);
}

__aicore__ inline void ScmDataCopyND2NZ(__gm__ void* kfcMsgPtr)
{
    ASCENDC_ASSERT((g_coreType == AIC), { KERNEL_LOG(KERNEL_ERROR, "core type must be AIC"); });
    auto scmCopyParams = reinterpret_cast<__gm__ struct Gm2L1Nd2NzParams*>(kfcMsgPtr);
    auto& intriParams = scmCopyParams->intri;
    auto l1AddrDst = reinterpret_cast<__cbuf__ void*>(scmCopyParams->dst);
#if ASCENDC_CPU_DEBUG
    uint8_t* aicBaseTSCMAddr = GetTPipePtr()->GetBaseAddr(int8_t(TPosition::TSCM));
    ASCENDC_ASSERT((aicBaseTSCMAddr != nullptr), { KERNEL_LOG(KERNEL_ERROR, "aicBaseTSCMAddr can not be nullptr"); });
    uint64_t l1AbsAddr = reinterpret_cast<uint64_t>(l1AddrDst);
    l1AddrDst = l1AbsAddr + reinterpret_cast<uint8_t*>(aicBaseTSCMAddr);
    ASCENDC_ASSERT((l1AbsAddr < TOTAL_L1_SIZE), {
        KERNEL_LOG(KERNEL_ERROR, "l1AbsAddr is 0x%lx, which should be in range of [0, %u)", l1AbsAddr, TOTAL_L1_SIZE);
    });
#endif

    if (scmCopyParams->dataTypeLen == 2) {
        copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ half*)l1AddrDst, (__gm__ half*)scmCopyParams->src, 0,
            intriParams.ndNum, intriParams.nValue, intriParams.dValue, intriParams.srcNdMatrixStride,
            intriParams.srcDValue, intriParams.dstNzC0Stride, intriParams.dstNzNStride, intriParams.dstNzMatrixStride);
    } else if (scmCopyParams->dataTypeLen == 4) {
        copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ float*)l1AddrDst, (__gm__ float*)scmCopyParams->src, 0,
            intriParams.ndNum, intriParams.nValue, intriParams.dValue, intriParams.srcNdMatrixStride,
            intriParams.srcDValue, intriParams.dstNzC0Stride, intriParams.dstNzNStride, intriParams.dstNzMatrixStride);
    } else {
        ASCENDC_ASSERT((scmCopyParams->dataTypeLen == 1), {
            KERNEL_LOG(KERNEL_ERROR, "type len is %d bytes, which should only be 1/2/4 bytes",
                scmCopyParams->dataTypeLen);
        });
        copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ int8_t*)l1AddrDst, (__gm__ int8_t*)scmCopyParams->src, 0,
            intriParams.ndNum, intriParams.nValue, intriParams.dValue, intriParams.srcNdMatrixStride,
            intriParams.srcDValue, intriParams.dstNzC0Stride, intriParams.dstNzNStride, intriParams.dstNzMatrixStride);
    }
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
    SetFlag<HardEvent::MTE2_MTE1>(eventID);
    WaitFlag<HardEvent::MTE2_MTE1>(eventID);
}

template <typename T>
__aicore__ inline void DataCopyGM2UBSingleImpl(__ubuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams,
    const int copyTime, const int computeNum)
{
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(dst)) % ONE_BLK_SIZE == 0),
        "dst address should be 32B aligned \n");
    const uint16_t &nValue = intriParams.nValue;
    const uint16_t& dValue = intriParams.dValue;
    const uint16_t& computeLen = computeNum * sizeof(T);
    const uint16_t& c0Count = DEFAULT_C0_SIZE / sizeof(T);
    const uint16_t& maxC0Count = MAX_REPEAT_TIMES * c0Count;
    const uint16_t& maxdValue = MAX_REPEAT_TIMES * dValue;
    const uint16_t& dstNzNStride = intriParams.dstNzNStride;
    const uint16_t& dstNzC0Stride = intriParams.dstNzC0Stride;
    const uint16_t& repeatCount = nValue / MAX_REPEAT_TIMES;
    const uint16_t& repeatTail = nValue % MAX_REPEAT_TIMES;
    const uint16_t& srcCopyStartOffset = copyTime * c0Count;
    const uint16_t& dstCopyStartOffset = copyTime * dstNzC0Stride * (DEFAULT_C0_SIZE / sizeof(T));
    DataCopyExtParams copyParams = { MAX_REPEAT_TIMES, (uint32_t)computeLen,
        (uint32_t)(intriParams.srcDValue * sizeof(T) - computeLen),
        (uint32_t)((dstNzNStride - (uint16_t)DEFAULT_C0_SIZE) / (uint16_t)DEFAULT_C0_SIZE), 0 };
    DataCopyPadExtParams<T> padParams;
    for (int repeatTime = 0; repeatTime < repeatCount; ++repeatTime) {
        DataCopyPadGm2UBImpl((__ubuf__ T*)(dst + dstCopyStartOffset + repeatTime * maxC0Count),
            (__gm__ T*)(src + srcCopyStartOffset + repeatTime * maxdValue), copyParams, padParams);
    }
    copyParams.blockCount = repeatTail;
    if (repeatTail != 0) {
        int dstOffset = (dstCopyStartOffset + repeatCount * MAX_REPEAT_TIMES * c0Count);
        int srcOffset = (srcCopyStartOffset + repeatCount * MAX_REPEAT_TIMES * dValue);
        DataCopyPadGm2UBImpl((__ubuf__ T*)(dst + dstOffset), (__gm__ T*)(src + srcOffset), copyParams, padParams);
    }
}

template <typename T>
__aicore__ inline void DataCopyGM2UBND2NZImpl(__ubuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(dst)) % ONE_BLK_SIZE == 0),
        "dst address should be 32B aligned \n");
    const uint16_t &ndNum = intriParams.ndNum;
    const uint16_t& dValue = intriParams.dValue;
    const uint16_t& srcNdMatrixStride = intriParams.srcNdMatrixStride;
    const uint16_t& srcDValue = intriParams.srcDValue;
    const uint16_t& dstNzC0Stride = intriParams.dstNzC0Stride;
    const uint16_t& dstNzNStride = intriParams.dstNzNStride;
    const uint16_t& dstNzMatrixStride = intriParams.dstNzMatrixStride;
    const uint16_t& c0Count = DEFAULT_C0_SIZE / sizeof(T);
    for (int index = 0; index < ndNum; ++index) {
        int16_t copyNum = (dValue + c0Count - 1) / c0Count;
        for (int copyTime = 0; copyTime < copyNum; ++copyTime) {
            int computeCount = (dValue >= (copyTime + 1) * c0Count) ? c0Count : (dValue % c0Count);
            DataCopyGM2UBSingleImpl(dst + dstNzMatrixStride, src + srcNdMatrixStride, intriParams, copyTime,
                computeCount);
        }
    }
}

template <typename T>
__aicore__ inline void DataCopyUB2GMNZ2NDImplBase(__gm__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t height,
    uint16_t width, uint16_t srcNStride, uint16_t dstDStride)
{
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(srcAddr) % ONE_BLK_SIZE == 0)),
        "src address should be 32B aligned \n");
    const uint16_t BLK_CNT_LIMIT = UINT12_MAX;                            // DataCopyPad max blockCount is 4095
    const uint16_t repeatTimes = height / BLK_CNT_LIMIT;
    const uint16_t tailBlock = height % BLK_CNT_LIMIT;
    const uint16_t widthBlkNum = (width + BLOCK_CUBE - 1) / BLOCK_CUBE;   // 1 nzMatrix has x columns of 16 elements

    for (uint16_t i = 0; i < widthBlkNum; ++i) {
        uint16_t num = (i != widthBlkNum -1) ? BLOCK_CUBE : (width - i * BLOCK_CUBE);  // num to copy in this block
        uint32_t blockLen = static_cast<uint32_t>(num * sizeof(T));
        uint32_t dstStride = static_cast<uint32_t>((dstDStride - num) * sizeof(T));
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            DataCopyPadUB2GMImpl(dstAddr + i * BLOCK_CUBE + j * BLK_CNT_LIMIT * dstDStride,
                srcAddr + i * srcNStride * BLOCK_CUBE + j * BLK_CNT_LIMIT * BLOCK_CUBE,
                {BLK_CNT_LIMIT, blockLen, 0, dstStride, 0});
        }
        if (tailBlock) {
            DataCopyPadUB2GMImpl(dstAddr + i * BLOCK_CUBE + repeatTimes * BLK_CNT_LIMIT * dstDStride,
                srcAddr + i * srcNStride * BLOCK_CUBE + repeatTimes * BLK_CNT_LIMIT * BLOCK_CUBE,
                {tailBlock, blockLen, 0, dstStride, 0});
        }
    }
}

template <typename T>
__aicore__ inline void DataCopyUB2GMNZ2NDImpl(__gm__ T* dst, __ubuf__ T* src, const Nz2NdParamsFull& intriParams)
{
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(src)) % ONE_BLK_SIZE == 0),
        "src address should be 32B aligned \n");
    ASCENDC_ASSERT((sizeof(T) == sizeof(int16_t) || sizeof(T) == sizeof(int32_t)),
        {KERNEL_LOG(KERNEL_ERROR, "DataCopy NZ2ND only suppports dtype B16 and B32");});
    const uint16_t ndNum = intriParams.ndNum;
    const uint16_t nValue = intriParams.nValue;
    const uint16_t dValue = intriParams.dValue;
    const uint16_t srcNdMatrixStride = intriParams.srcNdMatrixStride;
    const uint16_t srcNStride = intriParams.srcNStride;
    const uint16_t dstDStride = intriParams.dstDStride;
    const uint16_t dstNdMatrixStride = intriParams.dstNdMatrixStride;

    if (ndNum != 1 && nValue != 0) {
        ASCENDC_ASSERT(((srcNdMatrixStride * BLOCK_CUBE * BLOCK_CUBE) % (nValue * sizeof(T)) == 0),
            {KERNEL_LOG(KERNEL_ERROR,  "element num between nzMatrix head must be divisible by (nValue*sizeof(T))");});
    }
    ASCENDC_ASSERT((dValue % BLOCK_CUBE == 0), { KERNEL_LOG(KERNEL_ERROR,  "dValue must be divisible by 16"); });
    for (uint16_t i = 0; i < ndNum; ++i) {
        DataCopyUB2GMNZ2NDImplBase(dst + i * dstNdMatrixStride, src + i * srcNdMatrixStride * BLOCK_CUBE * BLOCK_CUBE,
            nValue, dValue, srcNStride, dstDStride);
    }
}

template <>
__aicore__ inline void DataCopyGM2UBSingleImpl(__ubuf__ float* dst, __gm__ float* src, const Nd2NzParams& intriParams,
    const int copyTime, const int computeNum)
{
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(dst)) % ONE_BLK_SIZE == 0),
        "dst address should be 32B aligned \n");
    const uint16_t &nValue = intriParams.nValue;
    const uint16_t& dValue = intriParams.dValue;
    const uint16_t& computeLen = computeNum * sizeof(float);
    const uint16_t& c0Count = BLOCK_CUBE;
    const uint16_t& maxC0Count = MAX_REPEAT_TIMES * c0Count;
    const uint16_t& maxdValue = MAX_REPEAT_TIMES * dValue;
    const uint16_t& dstNzNStride = intriParams.dstNzNStride;
    const uint16_t& dstNzC0Stride = intriParams.dstNzC0Stride;
    const uint16_t& repeatCount = nValue / MAX_REPEAT_TIMES;
    const uint16_t& repeatTail = nValue % MAX_REPEAT_TIMES;
    const uint16_t& srcCopyStartOffset = copyTime * c0Count;
    const uint16_t& dstCopyStartOffset = copyTime * dstNzC0Stride * (DEFAULT_C0_SIZE / sizeof(float));
    DataCopyExtParams copyParams = { MAX_REPEAT_TIMES, (uint32_t)computeLen,
        (uint32_t)(intriParams.srcDValue * sizeof(float) - computeLen),
        (uint32_t)((dstNzNStride * DEFAULT_C0_SIZE - (uint16_t)c0Count * sizeof(float)) / (uint16_t)DEFAULT_C0_SIZE),
        0 };
    DataCopyPadExtParams<float> padParams;
    if (computeNum < c0Count) {
        copyParams.dstStride = (c0Count - computeNum) * sizeof(float) / DEFAULT_C0_SIZE;
        padParams.paddingValue = 0;
    }

    for (int repeatTime = 0; repeatTime < repeatCount; ++repeatTime) {
        DataCopyPadGm2UBImpl((__ubuf__ float*)(dst + dstCopyStartOffset + repeatTime * maxC0Count),
            (__gm__ float*)(src + srcCopyStartOffset + repeatTime * maxdValue), copyParams, padParams);
    }
    copyParams.blockCount = repeatTail;
    if (repeatTail != 0) {
        int dstOffset = (dstCopyStartOffset + repeatCount * MAX_REPEAT_TIMES * c0Count);
        int srcOffset = (srcCopyStartOffset + repeatCount * MAX_REPEAT_TIMES * dValue);
        DataCopyPadGm2UBImpl((__ubuf__ float*)(dst + dstOffset), (__gm__ float*)(src + srcOffset), copyParams,
            padParams);
    }
}

template <>
__aicore__ inline void DataCopyGM2UBND2NZImpl(__ubuf__ float* dst, __gm__ float* src, const Nd2NzParams& intriParams)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::VECIN>(reinterpret_cast<uint64_t>(dst)) % ONE_BLK_SIZE == 0),
        "dst address should be 32B aligned \n");
    const uint16_t &ndNum = intriParams.ndNum;
    const uint16_t& dValue = intriParams.dValue;
    const uint16_t& srcNdMatrixStride = intriParams.srcNdMatrixStride;
    const uint16_t& srcDValue = intriParams.srcDValue;
    const uint16_t& dstNzC0Stride = intriParams.dstNzC0Stride;
    const uint16_t& dstNzNStride = intriParams.dstNzNStride;
    const uint16_t& dstNzMatrixStride = intriParams.dstNzMatrixStride;
    const uint16_t& c0Count = BLOCK_CUBE;
    for (int index = 0; index < ndNum; ++index) {
        int16_t copyNum = (dValue + c0Count - 1) / c0Count;
        for (int copyTime = 0; copyTime < copyNum; ++copyTime) {
            int computeCount = (dValue >= (copyTime + 1) * c0Count) ? c0Count : (dValue % c0Count);
            DataCopyGM2UBSingleImpl(dst + dstNzMatrixStride, src + srcNdMatrixStride, intriParams, copyTime,
                computeCount);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void DataCopyL0C2L1Impl(__cbuf__ T* dst, __cc__ U* src, const DataCopyCO12DstParams& intriParams)
{
    if ASCEND_IS_AIC {
        switch (intriParams.quantPre) {
            case QuantMode_t::F322F16:
                return copy_matrix_cc_to_cbuf(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::F322F16,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::F322BF16:
                return copy_matrix_cc_to_cbuf(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::F322BF16,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::DEQF16:
                return copy_matrix_cc_to_cbuf(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::DEQF16,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::VDEQF16:
                return copy_matrix_cc_to_cbuf(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::VDEQF16,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::QF322B8_PRE:
                return copy_matrix_cc_to_cbuf(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::QF322B8_PRE,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::VQF322B8_PRE:
                return copy_matrix_cc_to_cbuf(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::VQF322B8_PRE,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::REQ8:
                return copy_matrix_cc_to_cbuf(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::REQ8,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::VREQ8:
                return copy_matrix_cc_to_cbuf(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::VREQ8,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            default:
                ASCENDC_ASSERT(false, {KERNEL_LOG(KERNEL_ERROR, "Failed to check quantPre value in DataCopy from CO1 "
                    "to A1 / B1, supported values are F322F16 / F322BF16 / DEQF16 / VDEQF16 / QF322B8_PRE / "
                    "VQF322B8_PRE / REQ8 / VREQ8.");});
        }
    }
}

template <typename T, typename U>
__aicore__ inline void DataCopyL0C2GMImpl(__gm__ T* dst, __cc__ U* src, const DataCopyCO12DstParams& intriParams)
{
    if ASCEND_IS_AIC {
        ASCENDC_ASSERT((SupportType<Tuple<U, T>, Tuple<float, uint8_t>, Tuple<float, int8_t>, Tuple<float, half>,
            Tuple<float, bfloat16_t>, Tuple<float, float>, Tuple<int32_t, uint8_t>, Tuple<int32_t, int8_t>,
            Tuple<int32_t, half>, Tuple<int32_t, int16_t>, Tuple<int32_t, int32_t>>()), {KERNEL_LOG(KERNEL_ERROR,
            "Failed to check dtype in DataCopy from CO1 to GM, current api support dtype combination is src: float, "
            "dst: uint8_t / int8_t / half / bfloat16_t / float; src: int32_t, dst: uint8_t / int8_t / half / int16_t / "
            "int32_t.");});
        switch (intriParams.quantPre) {
            case QuantMode_t::NoQuant:
                return copy_matrix_cc_to_gm(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::NoQuant,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::F322F16:
                return copy_matrix_cc_to_gm(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::F322F16,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::F322BF16:
                return copy_matrix_cc_to_gm(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::F322BF16,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::DEQF16:
                return copy_matrix_cc_to_gm(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::DEQF16,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::VDEQF16:
                return copy_matrix_cc_to_gm(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::VDEQF16,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::QF322B8_PRE:
                return copy_matrix_cc_to_gm(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::QF322B8_PRE,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::VQF322B8_PRE:
                return copy_matrix_cc_to_gm(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::VQF322B8_PRE,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::REQ8:
                return copy_matrix_cc_to_gm(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::REQ8,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            case QuantMode_t::VREQ8:
                return copy_matrix_cc_to_gm(dst, src, intriParams.sid, intriParams.nSize, intriParams.mSize,
                    intriParams.dstStride, intriParams.srcStride, intriParams.unitFlag, QuantMode_t::VREQ8,
                    intriParams.reluPre, intriParams.channelSplit, intriParams.nz2ndEn);
            default:
                ASCENDC_ASSERT(false, {KERNEL_LOG(KERNEL_ERROR, "Failed to check quantPre value in DataCopy from CO1 "
                    "to GM, supported values are NoQuant / F322F16 / F322BF16 / DEQF16 / VDEQF16 / QF322B8_PRE / "
                    "VQF322B8_PRE / REQ8 / VREQ8.");});
        }
    }
}

template <typename T>
__aicore__ inline void DataCopyUB2L1Intf(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const DataCopyParams &intriParams)
{
    DataCopyUB2L1Impl((__cbuf__ PrimT<T>*)dstLocal.GetPhyAddr(), (__ubuf__ PrimT<T>*)srcLocal.GetPhyAddr(),
        intriParams);
}

template <typename T>
__aicore__ inline void DataCopyUB2L0CIntf(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    DataCopyUB2L0CImpl((__cc__ PrimT<T>*)dstLocal.GetPhyAddr(), (__ubuf__ PrimT<T>*)srcLocal.GetPhyAddr(),
        intriParams, enhancedParams);
}

#pragma begin_pipe(V)
template <typename T>
__aicore__ inline void DataCopyUB2UBIntf(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const DataCopyParams &intriParams)
{
    DataCopyUB2UBImpl((__ubuf__ PrimT<T>*)dstLocal.GetPhyAddr(), (__ubuf__ PrimT<T>*)srcLocal.GetPhyAddr(),
        intriParams);
}
#pragma end_pipe

template <typename T>
__aicore__ inline void DataCopyL12UBIntf(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const DataCopyParams &intriParams)
{
    DataCopyL12UBImpl((__ubuf__ PrimT<T>*)dstLocal.GetPhyAddr(), (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(),
        intriParams);
}

template <typename T>
__aicore__ inline void __inout_pipe__(MTE1) DataCopyL12L0CIntf(const LocalTensor<T> &dstLocal,
    const LocalTensor<T> &srcLocal, const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    DataCopyL12L0CImpl((__cc__ PrimT<T>*)dstLocal.GetPhyAddr(), (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(),
        intriParams, enhancedParams);
}

template <typename T>
__aicore__ inline void DataCopyL0C2UBIntf(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    DataCopyL0C2UBImpl((__ubuf__ PrimT<T>*)dstLocal.GetPhyAddr(), (__cc__ PrimT<T>*)srcLocal.GetPhyAddr(),
        intriParams, enhancedParams);
}

template <typename T>
__aicore__ inline __inout_pipe__(MTE1) void DataCopyL12BTIntf(const LocalTensor<T> &dstLocal,
    const LocalTensor<T> &srcLocal, const DataCopyParams &repeatParams)
{
    DataCopyL12BTImpl((uint64_t)dstLocal.GetPhyAddr(), (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), (uint16_t)0,
        repeatParams);
}

template <typename T>
__aicore__ inline __inout_pipe__(FIX) void DataCopyL12FBIntf(const LocalTensor<T> &dstLocal,
    const LocalTensor<T> &srcLocal, const DataCopyParams &repeatParams)
{
    DataCopyL12FBImpl((__fbuf__ PrimT<T>*)dstLocal.GetPhyAddr(), (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(),
        repeatParams);
}

template <typename T>
__aicore__ inline void DataCopyPadL12GMImpl(__gm__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "DataCopyPad from A1/B1/C1 to GM");
}

template <typename T>
__aicore__ inline void DataCopyPadL12GMImpl(__gm__ T* dst, __cbuf__ T* src, const DataCopyExtParams& intriParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "DataCopyPad from A1/B1/C1 to GM");
}

template <typename T>
__aicore__ inline void DataCopyPadGM2L1Impl(__cbuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams,
    const DataCopyPadParams& padParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "DataCopyPad from GM to A1/B1/C1");
}

template <typename T>
__aicore__ inline void DataCopyPadGM2L1Impl(__cbuf__ T* dst, __gm__ T* src, const DataCopyExtParams& intriParams,
    const DataCopyPadExtParams<T>& padParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "DataCopyPad from GM to A1/B1/C1");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H
