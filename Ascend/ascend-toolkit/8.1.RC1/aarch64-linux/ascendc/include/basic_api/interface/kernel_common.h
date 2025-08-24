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
 * \file kernel_common.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_COMMON_H
#define ASCENDC_KERNEL_COMMON_H

#include "kernel_reg.h"
#include "kernel_process_lock.h"

namespace AscendC {
class TPipe;
class KfcCommClient;

__aicore__ inline constexpr int16_t GetDataBlockSizeInBytes()
{
    return ONE_BLK_SIZE;
}
} // namespace AscendC

#if __CCE_AICORE__ == 220
#ifdef __DAV_C220_CUBE__
__BLOCK_LOCAL__ __inline__ AscendC::TPipe* g_cubeTPipePtr;
#elif defined(__DAV_C220_VEC__)
__BLOCK_LOCAL__ __inline__ AscendC::TPipe* g_vecTPipePtr;
#else
__BLOCK_LOCAL__ __inline__ AscendC::TPipe* g_tPipePtr;
#endif
#else
__BLOCK_LOCAL__ __inline__ AscendC::TPipe* g_tPipePtr;
#endif

#if __CCE_AICORE__ == 300 || defined(__DAV_M310__)
__BLOCK_LOCAL__ __inline__ uint64_t g_maskCount;
__BLOCK_LOCAL__ __inline__ half g_deqValue;
#endif
#if __CCE_AICORE__ == 220
__BLOCK_LOCAL__ __inline__ half g_deqValue;
#endif
__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_sysWorkspaceReserved;
__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_dumpWorkspaceReserved;
__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_hcclContextReserved[2];

#if defined(UT_TEST) || defined(ST_TEST)
__aicore__ AscendC::TPipe* GetTPipePtr();
#else
__aicore__ inline AscendC::TPipe* GetTPipePtr()
{
#if __CCE_AICORE__ == 220
#ifdef __DAV_C220_CUBE__
    return g_cubeTPipePtr;
#elif defined(__DAV_C220_VEC__)
    return g_vecTPipePtr;
#else
    return g_tPipePtr;
#endif
#else
    return g_tPipePtr;
#endif
}
#endif

namespace AscendC {
/* \brief the TensorTrait of tensor;
 * \note this struct contains primitive type of tensor;
 * info:
 * LiteType: the tensor's primitive type
 */
template <typename T>
struct TensorTrait {
    using LiteType = T;
};

template <typename T, MaskMode mode = MaskMode::NORMAL>
__aicore__ static inline void SetVectorMask(const uint64_t maskHigh, const uint64_t maskLow)
{
#if __CCE_AICORE__ == 300 || defined(__DAV_M310__)
    if (mode == MaskMode::COUNTER) {
        g_maskCount = maskLow;
    }
#endif
    SetVectorMaskImpl<T, mode>(maskHigh, maskLow);
}

template <typename T, MaskMode mode = MaskMode::NORMAL>
__aicore__ static inline void SetVectorMask(int32_t len)
{
#if __CCE_AICORE__ == 300 || defined(__DAV_M310__)
    g_maskCount = len;
#endif
    SetVectorMaskImpl<T, mode>(len);
}

__aicore__ inline void ResetMask()
{
#if __CCE_AICORE__ == 300 || defined(__DAV_M310__)
    g_maskCount = 0;
#endif
    ResetMaskImpl();
}

#if __CCE_AICORE__ >= 220
template <MemDsbT arg0>
__aicore__ inline void DataSyncBarrier()
{
    DataSyncBarrierImpl<arg0>();
}
#endif

template <HardEvent event> __aicore__ inline void SetFlag(int32_t eventID)
{
    if ASCEND_IS_AIC {
        if constexpr (event == HardEvent::MTE2_V || event == HardEvent::V_MTE2 || event == HardEvent::MTE3_V ||
            event == HardEvent::V_MTE3 || event == HardEvent::V_V || event == HardEvent::S_V ||
            event == HardEvent::V_S) {
            return;
        }
    }
    SetFlagImpl<event>(eventID);
}

template <HardEvent event> __aicore__ inline void WaitFlag(int32_t eventID)
{
    if ASCEND_IS_AIC {
        if constexpr (event == HardEvent::MTE2_V || event == HardEvent::V_MTE2 || event == HardEvent::MTE3_V ||
            event == HardEvent::V_MTE3 || event == HardEvent::V_V || event == HardEvent::S_V ||
            event == HardEvent::V_S) {
            return;
        }
    }
    WaitFlagImpl(event, eventID);
}

template <pipe_t pipe> __aicore__ inline void PipeBarrier()
{
    PipeBarrierImpl<pipe>();
}

#if (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 300)
template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dstTensor)
{
    DcciGMImpl<T, entireType, dcciDst>(const_cast<__gm__ T*>(dstTensor.GetPhyAddr()));
}

template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const LocalTensor<T>& dstTensor)
{
    DcciUBImpl<T, entireType, dcciDst>(const_cast<__ubuf__ T*>(dstTensor.GetPhyAddr()));
}
#endif

#if (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 200) || (__CCE_AICORE__ == 300)
template <typename T, CacheLine entireType>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dstTensor)
{
    DcciGMImpl<T, entireType>(const_cast<__gm__ T*>(dstTensor.GetPhyAddr()));
}
#endif

__aicore__ inline void SetMaskCount()
{
    SetMaskCountImpl();
}

__aicore__ inline void SetMaskNorm()
{
    SetMaskNormImpl();
}

__aicore__ inline void SetHF32Mode(bool hf32Mode)
{
    SetHF32ModeImpl(hf32Mode);
}

__aicore__ inline void SetHF32TransMode(bool hf32TransMode)
{
    SetHF32TransModeImpl(hf32TransMode);
}

__aicore__ inline void SetMMLayoutTransform(bool mmLayoutMode)
{
    SetMMLayoutTransformImpl(mmLayoutMode);
}

template <uint32_t index>
__aicore__ inline void SetHcclContext(__gm__ uint8_t* context)
{
    if constexpr (index > 1) {
        return;
    }
    g_hcclContextReserved[index] = context;
}

template <uint32_t index>
__aicore__ inline __gm__ uint8_t* __gm__ GetHcclContext(void)
{
    if constexpr (index > 1) {
        return nullptr;
    }
    return g_hcclContextReserved[index];
}

#if (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 200) || (__CCE_AICORE__ == 300)
template <typename T, typename U>
__aicore__ inline void SetAippFunctions(const GlobalTensor<T>& src0, AippInputFormat format, AippParams<U> config)
{
    SetAippFunctionsImpl<PrimT<T>, U>(const_cast<__gm__ PrimT<T>*>(src0.GetPhyAddr()), format, config);
}

template <typename T, typename U>
__aicore__ inline void SetAippFunctions(const GlobalTensor<T>& src0, const GlobalTensor<T>& src1,
    AippInputFormat format, AippParams<U> config)
{
    SetAippFunctionsImpl<PrimT<T>, U>(const_cast<__gm__ PrimT<T>*>(src0.GetPhyAddr()),
        const_cast<__gm__ PrimT<T>*>(src1.GetPhyAddr()), format, config);
}
#endif // (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 200) || (__CCE_AICORE__ == 300)
} // namespace AscendC

[[deprecated("NOTICE: SetDumpWorkSpacePtr has been deprecated and will be removed in the next version. "
        "Please do not use it!")]]
__aicore__ inline __gm__ uint8_t* __gm__ SetDumpWorkSpacePtr(__gm__ uint8_t* workspace)
{
    return g_dumpWorkspaceReserved = workspace;
}
[[deprecated("NOTICE: GetDumpWorkSpacePtr has been deprecated and will be removed in the next version. "
        "Please do not use it!")]]
__aicore__ inline __gm__ uint8_t* __gm__ GetDumpWorkSpacePtr()
{
    return g_dumpWorkspaceReserved;
}
#if defined(ASCENDC_CPU_DEBUG)
__aicore__ __gm__ uint8_t* __gm__ GetSysWorkSpacePtr();
__aicore__ void SetSysWorkSpacePtr(__gm__ uint8_t* workspace);
#else
__aicore__ inline __gm__ uint8_t* __gm__ GetSysWorkSpacePtr()
{
    return g_sysWorkspaceReserved;
}
[[deprecated(
    "NOTICE: SetSysWorkSpacePtr has been deprecated and will be removed in the next version.")]]
__aicore__ inline void SetSysWorkSpacePtr(__gm__ uint8_t* workspace)
{
    g_sysWorkspaceReserved = workspace;
}
#endif
#endif
