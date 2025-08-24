/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_common.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_COMMON_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_COMMON_IMPL_H

#include "kernel_pop_stack_buffer.h"
#include "softmax_common/softmax_common_utils.h"
#include "softmax_common/softmax_common_shape_process.h"
#include "softmax_common/softmax_tiling_func.h"
#include "softmax_common/softmax_common_broadcast.h"
#include "softmax_common/softmax_common_reduce.h"
#include "softmax_common/softmax_common_arithmetic.h"

namespace AscendC {

__aicore__ inline void CreateSpecialFormatMask(uint64_t& lowMask, const uint32_t& maskLen, const uint32_t& nzBlockCount)
{
    // create mask in "01111111 11111111 01111111 11111111" format
    // maskLen is 1-15
    ASCENDC_ASSERT((maskLen <= SOFTMAX_SHAPE_NZ_BASIC_COUNT),
                   { KERNEL_LOG(KERNEL_ERROR, "maskLen must less than 16"); });
    ASCENDC_ASSERT((nzBlockCount <= B32_BYTE_SIZE),
                   { KERNEL_LOG(KERNEL_ERROR, "nzBlockCount must less than 4"); });
    ASCENDC_ASSERT((nzBlockCount >= 1),
                   { KERNEL_LOG(KERNEL_ERROR, "nzBlockCount must large than 1"); });
    uint64_t defaultMask = 0xFFFF >> (SOFTMAX_SHAPE_NZ_BASIC_COUNT - maskLen); // logic shift right
    lowMask = defaultMask;

    for (uint32_t i = 0; i < nzBlockCount - 1; i++) {
        lowMask = lowMask << SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        lowMask = lowMask | defaultMask;
    }
}

__aicore__ inline void BinaryComputeWithSpecialMask(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, uint64_t mask[2], const uint32_t& lastBlockMaskLen, const uint32_t& splitCount,
    void (*func)(const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t*,
    const uint8_t, const BinaryRepeatParams&))
{
    uint32_t repeat = splitCount / FLOAT_REPEAT_SIZE;
    uint32_t tail = splitCount % FLOAT_REPEAT_SIZE;

    uint32_t repeatRange = repeat / MAX_REPEAT_TIMES;
    uint32_t repeatTail = repeat % MAX_REPEAT_TIMES;
    const auto offsetCount = MAX_REPEAT_TIMES * FLOAT_REPEAT_SIZE;
    uint32_t dstOffset = 0;
    uint32_t src0Offset = 0;
    uint32_t src1Offset = 0;

    for (uint32_t i = 0; i < repeatRange; i++) {
        func(dst[i * offsetCount], src0[i * offsetCount], src1[i * offsetCount], mask, MAX_REPEAT_TIMES,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    if (repeatTail != 0) {
        func(dst[repeatRange * offsetCount], src0[repeatRange * offsetCount], src1[repeatRange * offsetCount], mask,
            repeatTail, { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }

    if (tail != 0) {
        uint64_t tailMask[2] = { 0, 0 };
        CreateSpecialFormatMask(tailMask[0], lastBlockMaskLen, tail / SOFTMAX_SHAPE_NZ_BASIC_COUNT);
        func(dst[repeat * FLOAT_REPEAT_SIZE], src0[repeat * FLOAT_REPEAT_SIZE], src1[repeat * FLOAT_REPEAT_SIZE],
            tailMask, 1, { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
}

__aicore__ inline void UnaryComputeWithSpecialMask(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    uint64_t mask[2], const uint32_t& lastBlockMaskLen, const uint32_t& splitCount,
    void (*func)(const LocalTensor<float>&, const LocalTensor<float>&, uint64_t*, const uint8_t,
    const UnaryRepeatParams&))
{
    uint32_t repeat = splitCount / FLOAT_REPEAT_SIZE;
    uint32_t tail = splitCount % FLOAT_REPEAT_SIZE;

    uint32_t repeatRange = repeat / MAX_REPEAT_TIMES;
    uint32_t repeatTail = repeat % MAX_REPEAT_TIMES;
    const auto offsetCount = MAX_REPEAT_TIMES * FLOAT_REPEAT_SIZE;
    uint32_t dstOffset = 0;
    uint32_t src0Offset = 0;
    uint32_t src1Offset = 0;

    for (uint32_t i = 0; i < repeatRange; i++) {
        func(dst[i * offsetCount], src[i * offsetCount], mask, MAX_REPEAT_TIMES,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    if (repeatTail != 0) {
        func(dst[repeatRange * offsetCount], src[repeatRange * offsetCount], mask, repeatTail,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }

    if (tail != 0) {
        uint64_t tailMask[2] = { 0, 0 };
        CreateSpecialFormatMask(tailMask[0], lastBlockMaskLen, tail / SOFTMAX_SHAPE_NZ_BASIC_COUNT);
        func(dst[repeat * FLOAT_REPEAT_SIZE], src[repeat * FLOAT_REPEAT_SIZE], tailMask, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
}

}; // namespace AscendC
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_COMMON_IMPL_H