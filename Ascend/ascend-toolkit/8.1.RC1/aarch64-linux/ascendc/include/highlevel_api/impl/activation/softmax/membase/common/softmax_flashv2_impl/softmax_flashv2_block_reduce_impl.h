/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_flashv2_block_reduce_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_BLOCK_REDUCE_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_BLOCK_REDUCE_IMPL_H

namespace AscendC {
__aicore__ inline void SpecialBasicBlockMaxImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpbuffer, const uint8_t splitM, const uint8_t splitCeilM, const uint32_t splitK)
{
    if (splitK == DEFAULT_BLOCK_SIZE * HALF_FACTOR) { // reduce procedure : [8,512] -> [8,64] -> [8,8] ->[8,1]
        BlockReduceMax<float, false>(tmpbuffer, src, splitM * FLOAT_NUM_PER_BLK, MASK_PLACEHOLDER, 1, 1,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(tmpbuffer, tmpbuffer, splitM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(dst, tmpbuffer, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    } else { // reduce procedure : [8,1024] -> [8,128] -> [8,16] -> [8,8] ->[8,1]
        BlockReduceMax<float, false>(tmpbuffer, src, HALF_FACTOR * FLOAT_REPEAT_SIZE, MASK_PLACEHOLDER, 1, 1,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(tmpbuffer, tmpbuffer, HALF_FACTOR * FLOAT_NUM_PER_BLK, MASK_PLACEHOLDER, 1, 1,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        Max<float, false>(tmpbuffer, tmpbuffer, tmpbuffer[FLOAT_NUM_PER_BLK], 1, 1,
            { 1, HALF_FACTOR, HALF_FACTOR, DEFAULT_REPEAT_STRIDE, B16_DATA_NUM_PER_BLOCK, B16_DATA_NUM_PER_BLOCK });
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(dst, tmpbuffer, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    }
}

__aicore__ inline void SpecialBasicBlockAddImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpbuffer, const uint8_t splitM, const uint8_t splitCeilM, const uint32_t splitK)
{
    if (splitK == DEFAULT_BLOCK_SIZE * HALF_FACTOR) {  // reduce procedure : [8,512] -> [8,64] -> [8,8] ->[8,1]
        BlockReduceSum<float, false>(tmpbuffer, src, splitM * FLOAT_NUM_PER_BLK, MASK_PLACEHOLDER, 1, 1,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(tmpbuffer, tmpbuffer, splitM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(dst, tmpbuffer, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    } else { // reduce procedure : [8,1024] -> [8,128] -> [8,16] -> [8,8] ->[8,1]
        BlockReduceSum<float, false>(tmpbuffer, src, HALF_FACTOR * FLOAT_REPEAT_SIZE, MASK_PLACEHOLDER, 1, 1,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(tmpbuffer, tmpbuffer, HALF_FACTOR * FLOAT_NUM_PER_BLK, MASK_PLACEHOLDER, 1, 1,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        Add<float, false>(tmpbuffer, tmpbuffer, tmpbuffer[FLOAT_NUM_PER_BLK], 1, 1,
            { 1, HALF_FACTOR, HALF_FACTOR, DEFAULT_REPEAT_STRIDE, B16_DATA_NUM_PER_BLOCK, B16_DATA_NUM_PER_BLOCK });
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(dst, tmpbuffer, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    }
}

__aicore__ inline void BasicBlockReduceMaxImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpBuffer, const uint32_t splitBlock, const uint32_t splitM, const uint32_t splitK)
{
    const uint8_t splitCeilM = (uint8_t)(DivCeil(splitM, FLOAT_NUM_PER_BLK));
    if (splitK == DEFAULT_BLOCK_SIZE * HALF_FACTOR || splitK == SOFTMAX_SPECIAL_BASICBLOCK_LEN) {
        SpecialBasicBlockMaxImpl(dst, src, tmpBuffer, (uint8_t)splitM, splitCeilM, splitK);
    } else {
        if (splitBlock == 1) {
            BlockReduceMax<float, false>(tmpBuffer, src, (uint8_t)splitM, MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        } else if (splitK > DEFAULT_BLOCK_SIZE * HALF_FACTOR) {
            BigBlockReduceMax(tmpBuffer, src, splitBlock, splitM, splitK);
            PipeBarrier<PIPE_V>();
            BlockReduceMax<float, false>(tmpBuffer, tmpBuffer, (uint8_t)splitM, MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        } else {
            uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (splitK / FLOAT_REPEAT_SIZE));
            BasicBlockMaxImpl(tmpBuffer, src, (uint8_t)splitM, offset, splitBlock);
            PipeBarrier<PIPE_V>();
            BlockReduceMax<float, false>(tmpBuffer, tmpBuffer, (uint8_t)splitM, MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        }
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(dst, tmpBuffer, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    }
}
__aicore__ inline void BasicBlockReduceSumImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpBuffer, const uint32_t splitBlock, const uint32_t splitM, const uint32_t splitK)
{
    const uint8_t splitCeilM = (uint8_t)(DivCeil(splitM, FLOAT_NUM_PER_BLK));
    if (splitK == DEFAULT_BLOCK_SIZE * HALF_FACTOR || splitK == SOFTMAX_SPECIAL_BASICBLOCK_LEN) {
        SpecialBasicBlockAddImpl(dst, src, tmpBuffer, (uint8_t)splitM, splitCeilM, splitK);
    } else {
        if (splitBlock == 1) {
            BlockReduceSum<float, false>(tmpBuffer, src, (uint8_t)splitM, MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        } else if (splitK > DEFAULT_BLOCK_SIZE * HALF_FACTOR) {
            BigBlockReduceSum(tmpBuffer, src, splitBlock, splitM, splitK);
            PipeBarrier<PIPE_V>();
            BlockReduceSum<float, false>(tmpBuffer, tmpBuffer, (uint8_t)splitM, MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        } else {
            uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (splitK / FLOAT_REPEAT_SIZE));
            BasicBlockAddImpl(tmpBuffer, src, (uint8_t)splitM, offset, splitBlock);
            PipeBarrier<PIPE_V>();
            BlockReduceSum<float, false>(tmpBuffer, tmpBuffer, (uint8_t)splitM, MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        }
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(dst, tmpBuffer, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    }
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_BLOCK_REDUCE_IMPL_H