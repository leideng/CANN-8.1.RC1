/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file softmax_flash_basic_block_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_BASIC_BLOCK_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_BASIC_BLOCK_IMPL_H

namespace AscendC {

template <typename T>
__aicore__ inline void SoftmaxFlashBasicBlock(const LocalTensor<T> &dst, const LocalTensor<T> &sumTensor,
    const LocalTensor<T> &maxTensor, const LocalTensor<T> &src, const LocalTensor<T> &expMaxTensor,
    const LocalTensor<T> &inSumTensor, const LocalTensor<T> &inMaxTensor, const LocalTensor<float> &workLocal,
    const SoftMaxTiling &tiling)
{
    const LocalTensor<float> &tmpBuffer0 = workLocal[0];
    const LocalTensor<float> &tmpBuffer1 = workLocal[tiling.splitSize];
    const LocalTensor<float> &tmpBuffer2 = workLocal[tiling.splitSize + tiling.splitSize];
    const LocalTensor<float> &tmpBuffer3 = workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize];
    const LocalTensor<float> &inSumTmp =
        workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize + tiling.reduceSize];
    const LocalTensor<float> &inMaxTmp =
        workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize + tiling.reduceSize + tiling.reduceSize];

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
    const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    const uint32_t halfSplitSize = tiling.splitSize / B16_BYTE_SIZE;
    BinaryRepeatParams binaryRepeatParams;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();

        Cast<float, half, false>(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_BLK_NUM, HALF_DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        Max<float, false>(tmpBuffer1, tmpBuffer0, tmpBuffer0[FLOAT_REPEAT_SIZE], MASK_PLACEHOLDER,
            (uint8_t)(tiling.splitM), { 1, 1, 1, DEFAULT_BLK_NUM, offset, offset });
        for (uint32_t i = 2; i < splitBlock; ++i) {
            PipeBarrier<PIPE_V>();
            Max<float, false>(tmpBuffer1, tmpBuffer1, tmpBuffer0[FLOAT_REPEAT_SIZE * i], MASK_PLACEHOLDER,
                (uint8_t)(tiling.splitM), { 1, 1, 1, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM, offset });
        }
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(tmpBuffer1, tmpBuffer1, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
            DEFAULT_BLK_NUM);
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(tmpBuffer3, tmpBuffer1, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_BLK_NUM);
        PipeBarrier<PIPE_V>();
        Brcb(tmpBuffer1[halfSplitSize], tmpBuffer3, splitCeilM, { HALF_FACTOR, DEFAULT_REPEAT_STRIDE * HALF_FACTOR });
        Brcb(tmpBuffer1[halfSplitSize + DEFAULT_BLK_NUM], tmpBuffer3, splitCeilM,
            { HALF_FACTOR, DEFAULT_REPEAT_STRIDE * HALF_FACTOR });
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < splitBlock; ++i) {
            Sub<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * i], tmpBuffer0[FLOAT_REPEAT_SIZE * i],
                tmpBuffer1[halfSplitSize], MASK_PLACEHOLDER, (uint8_t)(tiling.splitM),
                { 1, 1, 0, offset, offset, B16_BYTE_SIZE });
        }
        PipeBarrier<PIPE_V>();

        Exp<float, false>(tmpBuffer0, tmpBuffer0, MASK_PLACEHOLDER, (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE),
            { 1, 1, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM });
        PipeBarrier<PIPE_V>();
        Add<float, false>(tmpBuffer1, tmpBuffer0, tmpBuffer0[FLOAT_REPEAT_SIZE], MASK_PLACEHOLDER,
            (uint8_t)(tiling.splitM), { 1, 1, 1, DEFAULT_REPEAT_STRIDE, offset, offset });
        for (uint32_t i = 2; i < splitBlock; ++i) {
            PipeBarrier<PIPE_V>();
            Add<float, false>(tmpBuffer1, tmpBuffer1, tmpBuffer0[FLOAT_REPEAT_SIZE * i], MASK_PLACEHOLDER,
                (uint8_t)(tiling.splitM), { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, offset });
        }
        PipeBarrier<PIPE_V>();

        BlockReduceSum<float, false>(tmpBuffer1, tmpBuffer1, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
            DEFAULT_BLK_NUM);
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(tmpBuffer3, tmpBuffer1, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_BLK_NUM);
        PipeBarrier<PIPE_V>();
        Brcb(tmpBuffer1, tmpBuffer3, splitCeilM, { HALF_FACTOR, DEFAULT_REPEAT_STRIDE * HALF_FACTOR });
        Brcb(tmpBuffer1[DEFAULT_BLK_NUM], tmpBuffer3, splitCeilM, { HALF_FACTOR, DEFAULT_REPEAT_STRIDE * HALF_FACTOR });
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < splitBlock; ++i) {
            Div<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * i], tmpBuffer0[FLOAT_REPEAT_SIZE * i], tmpBuffer1,
                MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 2 });
        }
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.reduceSize);

        Cast<float, half, false>(inMaxTmp, inMaxTensor[offset2], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();

        Max<float, false>(tmpBuffer2, inMaxTmp, tmpBuffer1[halfSplitSize], MASK_PLACEHOLDER, reduceCeilValue,
            binaryRepeatParams);
        PipeBarrier<PIPE_V>();

        Cast<half, float, false>(maxTensor[offset2], tmpBuffer2, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_BLK_NUM });
        PipeBarrier<PIPE_V>();
        Sub<float, false>(tmpBuffer3, tmpBuffer1[halfSplitSize], tmpBuffer2, MASK_PLACEHOLDER, reduceCeilValue,
            binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Exp<float, false>(tmpBuffer3, tmpBuffer3, MASK_PLACEHOLDER, reduceCeilValue,
            { 1, 1, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM });
        Sub<float, false>(inMaxTmp, inMaxTmp, tmpBuffer2, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();

        Exp<float, false>(inMaxTmp, inMaxTmp, MASK_PLACEHOLDER, reduceCeilValue,
            { 1, 1, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM });

        Cast<float, half, false>(inSumTmp, inSumTensor[offset2], RoundMode::CAST_NONE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, DEFAULT_BLK_NUM, HALF_DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        Mul<float, false>(inMaxTmp, inMaxTmp, inSumTmp, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        Mul<float, false>(tmpBuffer3, tmpBuffer3, tmpBuffer1, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Add<float, false>(inSumTmp, inMaxTmp, tmpBuffer3, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Div<float, false>(inMaxTmp, inMaxTmp, inSumTmp, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(expMaxTensor[offset2], inMaxTmp, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_BLK_NUM });
        Cast<half, float, false>(sumTensor[offset2], inSumTmp, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, reduceCeilValue,
            { 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_BLK_NUM });
        Div<float, false>(tmpBuffer3, tmpBuffer3, inSumTmp, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
        for (uint32_t i = 0; i < splitBlock; ++i) {
            Mul<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * i], tmpBuffer0[FLOAT_REPEAT_SIZE * i], tmpBuffer3,
                MASK_PLACEHOLDER, (uint8_t)(tiling.reduceM), { 1, 1, 0, offset, offset, HALF_FACTOR });
        }
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_BLK_NUM });
    }
}

#if __CCE_AICORE__ == 220 && ASCENDC_CPU_DEBUG == 0
__aicore__ inline void SoftmaxFlashBasicBlockFloat(const LocalTensor<float> &dst, const LocalTensor<float> &sumTensor,
    const LocalTensor<float> &maxTensor, const LocalTensor<float> &src, const LocalTensor<float> &expMaxTensor,
    const LocalTensor<float> &inSumTensor, const LocalTensor<float> &inMaxTensor, const LocalTensor<float> &workLocal,
    const SoftMaxTiling &tiling)
{
    const LocalTensor<float> &tmpBuffer1 = workLocal[0];
    const LocalTensor<float> &tmpBuffer2 = workLocal[tiling.splitSize];
    const LocalTensor<float> &tmpBuffer3 = workLocal[tiling.splitSize + tiling.splitSize];
    const LocalTensor<float> &inSumTmp = workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize];
    const LocalTensor<float> &inMaxTmp =
        workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize + tiling.reduceSize];

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
    const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    const uint32_t halfRepeatNum = DEFAULT_REPEAT_STRIDE / HALF_FACTOR;
    const uint32_t halfSplitSize = tiling.splitSize / HALF_FACTOR;
    BinaryRepeatParams binaryRepeatParams;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        __ubuf__ float *tmpBufferAddr0 = (__ubuf__ float *)src[offset1].GetPhyAddr();
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();

        if (splitBlock == 1) {
            Copy<float, false>(tmpBuffer1, src[offset1], MASK_PLACEHOLDER, repeatTimes,
                { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        } else {
            Max<float, false>(tmpBuffer1, src[offset1], src[offset1 + FLOAT_REPEAT_SIZE], MASK_PLACEHOLDER,
                (uint8_t)(tiling.splitM), { 1, 1, 1, DEFAULT_REPEAT_STRIDE, offset, offset });
            for (uint32_t j = 2; j < splitBlock; ++j) {
                PipeBarrier<PIPE_V>();
                Max<float, false>(tmpBuffer1, tmpBuffer1, src[offset1 + FLOAT_REPEAT_SIZE * j], MASK_PLACEHOLDER,
                    (uint8_t)(tiling.splitM), { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, offset });
            }
        }

        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(tmpBuffer1, tmpBuffer1, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(tmpBuffer3, tmpBuffer1, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();

        Brcb(tmpBuffer2[halfSplitSize], tmpBuffer3, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Sub<float, false>(src[offset1 + FLOAT_REPEAT_SIZE * j], src[offset1 + FLOAT_REPEAT_SIZE * j],
                tmpBuffer2[halfSplitSize], MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();
        Exp<float, false>(src[offset1], src[offset1], MASK_PLACEHOLDER, (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE),
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();

        if (splitBlock == 1) {
            Copy<float, false>(tmpBuffer1, src[offset1], MASK_PLACEHOLDER, repeatTimes,
                { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        } else {
            Add<float, false>(tmpBuffer1, src[offset1], src[offset1 + FLOAT_REPEAT_SIZE], MASK_PLACEHOLDER,
                (uint8_t)(tiling.splitM), { 1, 1, 1, DEFAULT_REPEAT_STRIDE, offset, offset });
            for (uint32_t j = 2; j < splitBlock; ++j) {
                PipeBarrier<PIPE_V>();
                Add<float, false>(tmpBuffer1, tmpBuffer1, src[offset1 + FLOAT_REPEAT_SIZE * j], MASK_PLACEHOLDER,
                    (uint8_t)(tiling.splitM), { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, offset });
            }
        }

        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(tmpBuffer1, tmpBuffer1, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(tmpBuffer3, tmpBuffer1, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        Brcb(tmpBuffer1, tmpBuffer3, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Div<float, false>(src[offset1 + FLOAT_REPEAT_SIZE * j], src[offset1 + FLOAT_REPEAT_SIZE * j], tmpBuffer1,
                MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();

        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.reduceSize);

        Copy<float, false>(inMaxTmp, inMaxTensor[offset2], MASK_PLACEHOLDER, 1, { 1, 1, 1, 0 });
        PipeBarrier<PIPE_V>();
        Max<float, false>(tmpBuffer2, inMaxTmp, tmpBuffer2[halfSplitSize], MASK_PLACEHOLDER, reduceCeilValue,
            binaryRepeatParams);
        PipeBarrier<PIPE_V>();

        Copy<float, false>(maxTensor[offset2], tmpBuffer2, MASK_PLACEHOLDER, 1, { 1, 1, 1, 0 });

        PipeBarrier<PIPE_V>();
        Sub<float, false>(tmpBuffer3, tmpBuffer2[halfSplitSize], tmpBuffer2, MASK_PLACEHOLDER, reduceCeilValue,
            binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Exp<float, false>(tmpBuffer3, tmpBuffer3, MASK_PLACEHOLDER, reduceCeilValue,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        Sub<float, false>(inMaxTmp, inMaxTmp, tmpBuffer2, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Exp<float, false>(inMaxTmp, inMaxTmp, MASK_PLACEHOLDER, reduceCeilValue,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });

        Copy<float, false>(inSumTmp, inSumTensor[offset2], MASK_PLACEHOLDER, 1, { 1, 1, 1, 0 });

        PipeBarrier<PIPE_V>();
        Mul<float, false>(inMaxTmp, inMaxTmp, inSumTmp, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        Mul<float, false>(tmpBuffer3, tmpBuffer3, tmpBuffer1, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Add<float, false>(inSumTmp, inMaxTmp, tmpBuffer3, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Div<float, false>(inMaxTmp, inMaxTmp, inSumTmp, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();

        // copy once
        Copy<float, false>(expMaxTensor[offset2], inMaxTmp, MASK_PLACEHOLDER, 1, { 1, 1, 1, 0 });
        Copy<float, false>(sumTensor[offset2], inSumTmp, MASK_PLACEHOLDER, 1, { 1, 1, 1, 0 });
        Div<float, false>(tmpBuffer3, tmpBuffer3, inSumTmp, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();

        ResetMask();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Mul<float, false>(src[offset1 + FLOAT_REPEAT_SIZE * j], src[offset1 + FLOAT_REPEAT_SIZE * j], tmpBuffer3,
                MASK_PLACEHOLDER, (uint8_t)(tiling.reduceM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();
    }
}
#endif

#if __CCE_AICORE__ == 220 && ASCENDC_CPU_DEBUG == 0
__aicore__ inline void SoftmaxFlashBasicBlock(const LocalTensor<half> &dst, const LocalTensor<float> &sumTensor,
    const LocalTensor<float> &maxTensor, const LocalTensor<half> &src, const LocalTensor<half> &expMaxTensor,
    const LocalTensor<float> &inSumTensor, const LocalTensor<float> &inMaxTensor, const LocalTensor<float> &workLocal,
    const SoftMaxTiling &tiling)
{
    const LocalTensor<float> &tmpBuffer0 = workLocal[0];
    const LocalTensor<float> &tmpBuffer1 = workLocal[tiling.splitSize];
    const LocalTensor<float> &tmpBuffer2 = workLocal[tiling.splitSize + tiling.splitSize];
    const LocalTensor<float> &tmpBuffer3 = workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize];
    const LocalTensor<float> &inSumTmp =
        workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize + tiling.reduceSize];
    const LocalTensor<float> &inMaxTmp =
        workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize + tiling.reduceSize + tiling.reduceSize];

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
    const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    const uint32_t halfRepeatNum = DEFAULT_REPEAT_STRIDE / B16_BYTE_SIZE;
    const uint32_t halfSplitSize = tiling.splitSize / B16_BYTE_SIZE;
    BinaryRepeatParams binaryRepeatParams;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
        Cast<float, half, false>(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_REPEAT_STRIDE, halfRepeatNum });
        PipeBarrier<PIPE_V>();
        Max<float, false>(tmpBuffer1, tmpBuffer0, tmpBuffer0[FLOAT_REPEAT_SIZE], MASK_PLACEHOLDER,
            (uint8_t)(tiling.splitM), { 1, 1, 1, DEFAULT_REPEAT_STRIDE, offset, offset });
        for (uint32_t j = 2; j < splitBlock; ++j) {
            PipeBarrier<PIPE_V>();
            Max<float, false>(tmpBuffer1, tmpBuffer1, tmpBuffer0[FLOAT_REPEAT_SIZE * j], MASK_PLACEHOLDER,
                (uint8_t)(tiling.splitM), { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, offset });
        }
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(tmpBuffer1, tmpBuffer1, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(tmpBuffer3, tmpBuffer1, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();

        Brcb(tmpBuffer1[halfSplitSize], tmpBuffer3, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Sub<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer0[FLOAT_REPEAT_SIZE * j],
                tmpBuffer1[halfSplitSize], MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();
        Exp<float, false>(tmpBuffer0, tmpBuffer0, MASK_PLACEHOLDER, (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE),
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        Add<float, false>(tmpBuffer1, tmpBuffer0, tmpBuffer0[FLOAT_REPEAT_SIZE], MASK_PLACEHOLDER,
            (uint8_t)(tiling.splitM), { 1, 1, 1, DEFAULT_REPEAT_STRIDE, offset, offset });
        for (uint32_t j = 2; j < splitBlock; ++j) {
            PipeBarrier<PIPE_V>();
            Add<float, false>(tmpBuffer1, tmpBuffer1, tmpBuffer0[FLOAT_REPEAT_SIZE * j], MASK_PLACEHOLDER,
                (uint8_t)(tiling.splitM), { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, offset });
        }
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(tmpBuffer1, tmpBuffer1, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(tmpBuffer3, tmpBuffer1, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        Brcb(tmpBuffer1, tmpBuffer3, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Div<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer1,
                MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.reduceSize);

        Copy<float, false>(inMaxTmp, inMaxTensor[offset2], MASK_PLACEHOLDER, 1, { 1, 1, 1, 0 });
        PipeBarrier<PIPE_V>();
        Max<float, false>(tmpBuffer2, inMaxTmp, tmpBuffer1[halfSplitSize], MASK_PLACEHOLDER, reduceCeilValue,
            binaryRepeatParams);
        PipeBarrier<PIPE_V>();

        Copy<float, false>(maxTensor[offset2], tmpBuffer2, MASK_PLACEHOLDER, 1, { 1, 1, 1, 0 });

        PipeBarrier<PIPE_V>();
        Sub<float, false>(tmpBuffer3, tmpBuffer1[halfSplitSize], tmpBuffer2, MASK_PLACEHOLDER, reduceCeilValue,
            binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Exp<float, false>(tmpBuffer3, tmpBuffer3, MASK_PLACEHOLDER, reduceCeilValue,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        Sub<float, false>(inMaxTmp, inMaxTmp, tmpBuffer2, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Exp<float, false>(inMaxTmp, inMaxTmp, MASK_PLACEHOLDER, reduceCeilValue,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });

        Copy<float, false>(inSumTmp, inSumTensor[offset2], MASK_PLACEHOLDER, 1, { 1, 1, 1, 0 });

        PipeBarrier<PIPE_V>();
        Mul<float, false>(inMaxTmp, inMaxTmp, inSumTmp, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        Mul<float, false>(tmpBuffer3, tmpBuffer3, tmpBuffer1, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Add<float, false>(inSumTmp, inMaxTmp, tmpBuffer3, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Div<float, false>(inMaxTmp, inMaxTmp, inSumTmp, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();

        // src 32B copy to dst 64B copy twice
        Copy<float, false>(tmpBuffer1, inMaxTmp, MASK_PLACEHOLDER, B16_BYTE_SIZE, { B16_BYTE_SIZE, 1, 1, 0 });
        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.reduceSize * B16_BYTE_SIZE); // shape is m * 16
        Cast<half, float, false>(expMaxTensor[offset2 * HALF_FACTOR], tmpBuffer1, FLOAT2HALF_ROUND_MODE,
            MASK_PLACEHOLDER, reduceCeilValue, { 1, 1, halfRepeatNum, DEFAULT_REPEAT_STRIDE });
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.reduceSize);
        // copy once
        Copy<float, false>(sumTensor[offset2], inSumTmp, MASK_PLACEHOLDER, 1, { 1, 1, 1, 0 });
        Div<float, false>(tmpBuffer3, tmpBuffer3, inSumTmp, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();

        ResetMask();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Mul<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer3,
                MASK_PLACEHOLDER, (uint8_t)(tiling.reduceM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, halfRepeatNum, DEFAULT_REPEAT_STRIDE });
    }
}
#endif

}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_BASIC_BLOCK_IMPL_H