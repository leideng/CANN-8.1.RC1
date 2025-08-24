/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file inner_kernel_operator_vec_scatter_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_SCATTER_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_SCATTER_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_scatter_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_scatter_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_scatter_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_scatter_impl.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
/*
 * @ingroup scatter Level 0
 * @brief scatter element from dstLocal according to dstOffsetLocal
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] dstOffsetLocal input LocalTensor
 * @param [in] dstBaseAddr base address of dstLocal
 * @param [in] mask valid element count
 * @param [in] repeatTimes repeat times
 * @param [in] dstRepStride dst repeat stride
 */
template <typename T>
__aicore__ inline void Scatter(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint32_t>& dstOffsetLocal, const uint32_t dstBaseAddr, const uint64_t mask,
    const uint8_t repeatTimes, const uint8_t srcRepStride)
{
#if __CCE_AICORE__ >= 300
    ASCENDC_ASSERT((SupportType<T, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Scatter, current api support dtype combination is src and "
        "dst both: uint8 / int8 / half / bfloat16_t / uint16_t / int16_t / float / uint32_t / int32_t");});
#else
    ASCENDC_ASSERT((SupportType<T, half, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Scatter, current api support dtype combination is src and "
        "dst both: half / uint16_t / int16_t / float / uint32_t / int32_t");});
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFunScatter(dstLocal, srcLocal, dstOffsetLocal, dstBaseAddr, mask, repeatTimes, srcRepStride, "Scatter")) {
        ASCENDC_REPORT_CHECK_ERROR("Scatter", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    const uint32_t dstLength = dstLocal.GetSize();
    ScatterImpl((__ubuf__ T *)dstLocal.GetPhyAddr(), (__ubuf__ T *)srcLocal.GetPhyAddr(),
        (__ubuf__ uint32_t *)dstOffsetLocal.GetPhyAddr(), dstLength, dstBaseAddr, mask, repeatTimes, srcRepStride);
}

/*
 * @ingroup scatter Level 0
 * @brief scatter element from dstLocal according to dstOffsetLocal
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] dstOffsetLocal input LocalTensor
 * @param [in] dstBaseAddr base address of dstLocal
 * @param [in] mask valid element count(bit mode)
 * @param [in] repeatTimes repeat times
 * @param [in] dstRepStride dst repeat stride
 */
template <typename T>
__aicore__ inline void Scatter(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint32_t>& dstOffsetLocal, const uint32_t dstBaseAddr, const uint64_t mask[],
    const uint8_t repeatTimes, const uint8_t srcRepStride)
{
#if __CCE_AICORE__ >= 300
    ASCENDC_ASSERT((SupportType<T, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Scatter, current api support dtype combination is src and "
        "dst both: uint8 / int8 / half / bfloat16_t / uint16_t / int16_t / float / uint32_t / int32_t");});
#else
    ASCENDC_ASSERT((SupportType<T, half, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Scatter, current api support dtype combination is src and "
        "dst both: half / uint16_t / int16_t / float / uint32_t / int32_t");});
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFunScatter(dstLocal, srcLocal, dstOffsetLocal, dstBaseAddr, mask, repeatTimes, srcRepStride, "Scatter")) {
        ASCENDC_REPORT_CHECK_ERROR("Scatter", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    const uint32_t dstLength = dstLocal.GetSize();
    ScatterImpl((__ubuf__ T *)dstLocal.GetPhyAddr(), (__ubuf__ T *)srcLocal.GetPhyAddr(),
        (__ubuf__ uint32_t *)dstOffsetLocal.GetPhyAddr(), dstLength, dstBaseAddr, mask, repeatTimes, srcRepStride);
}

/*
 * @ingroup scatter Level 2
 * @brief scatter element from dstLocal according to dstOffsetLocal
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] dstOffsetLocal input LocalTensor
 * @param [in] dstBaseAddr base address of dstLocal
 * @param [in] count element count
 */
template <typename T>
__aicore__ inline void Scatter(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint32_t>& dstOffsetLocal, const uint32_t dstBaseAddr, const uint32_t count)
{
#if __CCE_AICORE__ >= 300
    ASCENDC_ASSERT((SupportType<T, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Scatter, current api support dtype combination is src and "
        "dst both: uint8 / int8 / half / bfloat16_t / uint16_t / int16_t / float / uint32_t / int32_t");});
#else
    ASCENDC_ASSERT((SupportType<T, half, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Scatter, current api support dtype combination is src and "
        "dst both: half / uint16_t / int16_t / float / uint32_t / int32_t");});
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFunScatter(dstLocal, srcLocal, dstOffsetLocal, dstBaseAddr, count, "Scatter")) {
        ASCENDC_REPORT_CHECK_ERROR("Scatter", KernelFuncType::NONE_MODE);
    }
#endif
    #if __CCE_AICORE__ >= 300
    ScatterImpl((__ubuf__ T *)dstLocal.GetPhyAddr(), (__ubuf__ T *)srcLocal.GetPhyAddr(),
        (__ubuf__ uint32_t *)dstOffsetLocal.GetPhyAddr(), dstBaseAddr, count);
    #else
    uint32_t elementCountSingleRepeat;
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        elementCountSingleRepeat = 128;
    } else {
        elementCountSingleRepeat = 64;
    }
    const uint32_t elementCountTail = count % elementCountSingleRepeat;
    const uint8_t repeatTimes = count / elementCountSingleRepeat;
    if (repeatTimes > 0) {
        Scatter(dstLocal, srcLocal, dstOffsetLocal, dstBaseAddr, (uint64_t)elementCountSingleRepeat, repeatTimes,
            DEFAULT_REPEAT_STRIDE);
    }
    if (elementCountTail > 0) {
        const uint32_t offset = count - elementCountTail;
        Scatter(dstLocal, srcLocal[offset], dstOffsetLocal[offset], dstBaseAddr, (uint64_t)elementCountTail, 1,
            DEFAULT_REPEAT_STRIDE);
    }
    #endif
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_SCATTER_INTERFACE_H