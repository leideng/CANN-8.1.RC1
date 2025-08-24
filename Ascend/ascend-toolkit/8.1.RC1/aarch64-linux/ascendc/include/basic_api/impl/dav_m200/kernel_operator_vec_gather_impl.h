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
 * \file kernel_operator_vec_gather_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
#include "kernel_struct_gather.h"

namespace AscendC {
/* **************************************************************************************************
 * Gather                                             *
 * ************************************************************************************************* */
template <typename T>
[[deprecated("NOTICE: Gatherb is not deprecated. Currently, Gatherb is an unsupported API on current device. "
             "Please check your code!")]]
__aicore__ inline void GatherbImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint32_t* offset,
    const uint32_t srcLength, const uint8_t repeatTimes, const GatherRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Gatherb");
}

template <typename T>
__aicore__ inline void GatherImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ uint32_t* srcOffsetLocal,
    const uint32_t srcLength, const uint32_t srcBaseAddr, const uint64_t mask, const uint8_t repeatTimes,
    const uint16_t dstRepStride)
{
    uint32_t offsetAddr = (uint64_t)srcLocal + srcBaseAddr;
#if ASCENDC_CPU_DEBUG
    uint64_t cpuAddr = (uint64_t)srcLocal + srcBaseAddr;
    SetModelGatherbSrc0Tensor(cpuAddr, srcLength);
#endif
    uint64_t config = 0;
    config |= (uint64_t(offsetAddr & 0xFFFFFFFF));
    config |= (uint64_t(repeatTimes & 0xFF) << 56);
    config |= (uint64_t(dstRepStride & 0xFF) << 32);

    AscendCUtils::SetMask<T>(mask);
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        vgather((__ubuf__ uint16_t *)dstLocal, srcOffsetLocal, config);
    } else {
        vgather((__ubuf__ uint32_t *)dstLocal, srcOffsetLocal, config);
    }
}

template <typename T>
__aicore__ inline void GatherImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ uint32_t* srcOffsetLocal,
    const uint32_t srcLength, const uint32_t srcBaseAddr, const uint64_t mask[], const uint8_t repeatTimes,
    const uint16_t dstRepStride)
{
    uint32_t offsetAddr = (uint64_t)srcLocal + srcBaseAddr;
#if ASCENDC_CPU_DEBUG
    uint64_t cpuAddr = (uint64_t)srcLocal + srcBaseAddr;
    SetModelGatherbSrc0Tensor(cpuAddr, srcLength);
#endif
    uint64_t config = 0;
    config |= (uint64_t(offsetAddr & 0xFFFFFFFF));
    config |= (uint64_t(repeatTimes & 0xFF) << 56);
    config |= (uint64_t(dstRepStride & 0xFF) << 32);

    AscendCUtils::SetMask<T>(mask[1], mask[0]);
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        vgather((__ubuf__ uint16_t *)dstLocal, srcOffsetLocal, config);
    } else {
        vgather((__ubuf__ uint32_t *)dstLocal, srcOffsetLocal, config);
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H