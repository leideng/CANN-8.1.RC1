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
 * \file inner_kernel_operator_vec_gather_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_GATHER_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_GATHER_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"
#include "kernel_struct_gather.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_gather_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_gather_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_gather_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_gather_impl.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
/*
 * @ingroup gatherb Level 0
 * @brief this function fetches N addresses from offsetLocal,then accesses these N addresses(plus the src0Local address)
 * @brief to get N 32Byte block, and finally writes these N blocks into dstLocal.
 * @brief gather element in the uint of block
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] offsetLocal input LocalTensor
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 */
template <typename T>
__aicore__ inline void Gatherb(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<uint32_t>& offsetLocal, const uint8_t repeatTimes, const GatherRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncGatherb(dstLocal, src0Local, offsetLocal, repeatTimes, repeatParams, "Gatherb")) {
        ASCENDC_REPORT_CHECK_ERROR("Gatherb", KernelFuncType::NONE_MODE);
    }
#endif
    uint32_t srcLength = src0Local.GetSize();
    GatherbImpl((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(),
        (__ubuf__ uint32_t*)offsetLocal.GetPhyAddr(), srcLength, repeatTimes, repeatParams);
}

/*
 * @ingroup gather Level 0
 * @brief gather element from srcLocal according to srcOffsetLocal
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] srcOffsetLocal input LocalTensor
 * @param [in] srcBaseAddr base address of srcLocal
 * @param [in] mask valid element count
 * @param [in] repeatTimes repeat times
 * @param [in] dstRepStride dst repeat stride
 */
template <typename T>
__aicore__ inline void Gather(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint32_t>& srcOffsetLocal, const uint32_t srcBaseAddr, const uint64_t mask,
    const uint8_t repeatTimes, const uint16_t dstRepStride)
{
#if __CCE_AICORE__ >= 300
    ASCENDC_ASSERT((SupportType<T, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Gather, current api support dtype combination is src and "
        "dst both: uint8 / int8 / half / bfloat16_t / uint16_t / int16_t / float / uint32_t / int32_t");});
#elif __CCE_AICORE__ >= 220
    ASCENDC_ASSERT((SupportType<T, half, bfloat16_t, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Gather, current api support dtype combination is src and "
        "dst both: half / bfloat16_t / uint16_t / int16_t / float / uint32_t / int32_t");});
#else
    ASCENDC_ASSERT((SupportType<T, half, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Gather, current api support dtype combination is src and "
        "dst both: half / uint16_t / int16_t / float / uint32_t / int32_t");});
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncGather(dstLocal, srcLocal, srcOffsetLocal, srcBaseAddr, mask, repeatTimes, dstRepStride, "Gather")) {
        ASCENDC_REPORT_CHECK_ERROR("Gather", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    const uint32_t srcLength = srcLocal.GetSize();
    GatherImpl((__ubuf__ T *)dstLocal.GetPhyAddr(), (__ubuf__ T *)srcLocal.GetPhyAddr(),
        (__ubuf__ uint32_t *)srcOffsetLocal.GetPhyAddr(), srcLength, srcBaseAddr, mask, repeatTimes, dstRepStride);
}

/*
 * @ingroup gather Level 0
 * @brief gather element from srcLocal according to srcOffsetLocal
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] srcOffsetLocal input LocalTensor
 * @param [in] srcBaseAddr base address of srcLocal
 * @param [in] mask valid element count(bit mode)
 * @param [in] repeatTimes repeat times
 * @param [in] dstRepStride dst repeat stride
 */
template <typename T>
__aicore__ inline void Gather(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint32_t>& srcOffsetLocal, const uint32_t srcBaseAddr, const uint64_t mask[],
    const uint8_t repeatTimes, const uint16_t dstRepStride)
{
#if __CCE_AICORE__ >= 300
    ASCENDC_ASSERT((SupportType<T, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Gather, current api support dtype combination is src and "
        "dst both: uint8 / int8 / half / bfloat16_t / uint16_t / int16_t / float / uint32_t / int32_t");});
#elif __CCE_AICORE__ >= 220
    ASCENDC_ASSERT((SupportType<T, half, bfloat16_t, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Gather, current api support dtype combination is src and "
        "dst both: half / bfloat16_t / uint16_t / int16_t / float / uint32_t / int32_t");});
#else
    ASCENDC_ASSERT((SupportType<T, half, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Gather, current api support dtype combination is src and "
        "dst both: half / uint16_t / int16_t / float / uint32_t / int32_t");});
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncGather(dstLocal, srcLocal, srcOffsetLocal, srcBaseAddr, mask, repeatTimes, dstRepStride, "Gather")) {
        ASCENDC_REPORT_CHECK_ERROR("Gather", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    const uint32_t srcLength = srcLocal.GetSize();
    GatherImpl((__ubuf__ T *)dstLocal.GetPhyAddr(), (__ubuf__ T *)srcLocal.GetPhyAddr(),
        (__ubuf__ uint32_t *)srcOffsetLocal.GetPhyAddr(), srcLength, srcBaseAddr, mask, repeatTimes, dstRepStride);
}

/*
 * @ingroup gather Level 2
 * @brief gather element from srcLocal according to srcOffsetLocal
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] srcOffsetLocal input LocalTensor
 * @param [in] srcBaseAddr base address of srcLocal
 * @param [in] count element count
 */
template <typename T>
__aicore__ inline void Gather(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint32_t>& srcOffsetLocal, const uint32_t srcBaseAddr, const uint32_t count)
{
#if __CCE_AICORE__ >= 300
    ASCENDC_ASSERT((SupportType<T, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Gather, current api support dtype combination is src and "
        "dst both: uint8 / int8 / half / bfloat16_t / uint16_t / int16_t / float / uint32_t / int32_t");});
    GatherImpl((__ubuf__ T *)dstLocal.GetPhyAddr(), (__ubuf__ T *)srcLocal.GetPhyAddr(),
        (__ubuf__ uint32_t *)srcOffsetLocal.GetPhyAddr(), srcBaseAddr, count);
#else
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncGather(dstLocal, srcLocal, srcOffsetLocal, srcBaseAddr, count, "Gather")) {
        ASCENDC_REPORT_CHECK_ERROR("Gather", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    uint32_t elementCountSingleRepeat;
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        elementCountSingleRepeat = 128;
    } else {
        elementCountSingleRepeat = 64;
    }
    const uint32_t elementCountTail = count % elementCountSingleRepeat;
    const uint8_t repeatTimes = count / elementCountSingleRepeat;
    if (repeatTimes > 0) {
        Gather(dstLocal, srcLocal, srcOffsetLocal, srcBaseAddr, (uint64_t)elementCountSingleRepeat, repeatTimes,
            DEFAULT_REPEAT_STRIDE);
    }
    if (elementCountTail > 0) {
        const uint32_t offset = count - elementCountTail;
        Gather(dstLocal[offset], srcLocal, srcOffsetLocal[offset], srcBaseAddr, (uint64_t)elementCountTail, 1,
            DEFAULT_REPEAT_STRIDE);
    }
#endif
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_GATHER_INTERFACE_H