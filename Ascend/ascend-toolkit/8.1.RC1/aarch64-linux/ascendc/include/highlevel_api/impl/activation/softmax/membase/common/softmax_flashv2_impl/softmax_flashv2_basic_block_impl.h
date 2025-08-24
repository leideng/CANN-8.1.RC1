/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_flashv2_basic_block_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_BASIC_BLOCK_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_BASIC_BLOCK_IMPL_H

#include "softmax_flashv2_block_reduce_impl.h"

namespace AscendC {
__aicore__ inline void SetWaitFlagVToS()
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

__aicore__ inline void SoftmaxFlashV2BasicBlockImpl(const LocalTensor<half>& dst, const LocalTensor<half>& expSumTensor,
    const LocalTensor<half>& maxTensor, const LocalTensor<half>& src, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<half>& inExpSumTensor, const LocalTensor<half>& inMaxTensor, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitSize];
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize + tiling.reduceSize];
    const LocalTensor<float>& tmpBuffer3 =
        workLocal[tiling.splitSize + tiling.reduceSize + tiling.splitM * FLOAT_REPEAT_SIZE / B16_BYTE_SIZE];

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    // tiling will ensure repeatTimes large than 16
    uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
    const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    BinaryRepeatParams binaryRepeatParams;
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        Cast<float, half, false>(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        BasicBlockReduceMaxImpl(tmpBuffer3, tmpBuffer0, tmpBuffer1, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        SetWaitFlagVToS();
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(tmpBuffer1[FLOAT_REPEAT_SIZE * j * HALF_FACTOR], tmpBuffer3[FLOAT_NUM_PER_BLK * j],
                HALF_FACTOR * DEFAULT_REPEAT_STRIDE);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(tmpBuffer1, tmpBuffer3, splitCeilM, { B16_BYTE_SIZE, B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE });
        Brcb(tmpBuffer1[DEFAULT_REPEAT_STRIDE], tmpBuffer3, splitCeilM,
            { B16_BYTE_SIZE, B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE });
#endif

        PipeBarrier<PIPE_V>();
        Cast<float, half, false>(tmpBuffer2, inMaxTensor[offset2], RoundMode::CAST_NONE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        Max<float, false>(tmpBuffer3, tmpBuffer2, tmpBuffer1, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();

        Cast<half, float, false>(maxTensor[offset2], tmpBuffer3, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });

        // expmax = exp(inmax - max)
        Sub<float, false>(tmpBuffer2, tmpBuffer2, tmpBuffer3, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Exp<float, false>(tmpBuffer2, tmpBuffer2, MASK_PLACEHOLDER, reduceCeilValue,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(expMaxTensor[offset2], tmpBuffer2, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < splitBlock; ++i) {
            Sub<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * i], tmpBuffer0[FLOAT_REPEAT_SIZE * i], tmpBuffer3,
                MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, B16_BYTE_SIZE });
        }

        PipeBarrier<PIPE_V>();
        Exp<float, false>(tmpBuffer0, tmpBuffer0, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });

        BasicBlockReduceSumImpl(tmpBuffer3, tmpBuffer0, tmpBuffer1, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        SetWaitFlagVToS();
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(tmpBuffer1[FLOAT_REPEAT_SIZE * j * HALF_FACTOR], tmpBuffer3[FLOAT_NUM_PER_BLK * j],
                HALF_FACTOR * DEFAULT_REPEAT_STRIDE);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(tmpBuffer1, tmpBuffer3, splitCeilM, { B16_BYTE_SIZE, B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE });
        Brcb(tmpBuffer1[DEFAULT_REPEAT_STRIDE], tmpBuffer3, splitCeilM,
            { B16_BYTE_SIZE, B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE });
#endif

        PipeBarrier<PIPE_V>();
        // update sum = expmax * insum + sum
        Cast<float, half, false>(tmpBuffer3, inExpSumTensor[offset2], RoundMode::CAST_NONE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        Mul<float, false>(tmpBuffer3, tmpBuffer2, tmpBuffer3, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Add<float, false>(tmpBuffer3, tmpBuffer3, tmpBuffer1, MASK_PLACEHOLDER, reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(expSumTensor[offset2], tmpBuffer3, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
}

__aicore__ inline void SoftmaxFlashV2BasicBlockImpl(const LocalTensor<float>& dst,
    const LocalTensor<float>& expSumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<float>& src,
    const LocalTensor<float>& expMaxTensor, const LocalTensor<float>& inExpSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling)
{
    const LocalTensor<float>& tmpBuffer1 = workLocal;
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitM * FLOAT_REPEAT_SIZE];
    const LocalTensor<float>& tmpBuffer3 = workLocal[tiling.splitM * FLOAT_REPEAT_SIZE / B16_BYTE_SIZE];

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
    const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    BinaryRepeatParams binaryRepeatParams;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        PipeBarrier<PIPE_V>();
        BasicBlockReduceMaxImpl(tmpBuffer2, src[offset1], tmpBuffer1, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();

#if __CCE_AICORE__ == 200
        SetWaitFlagVToS();
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(tmpBuffer1[FLOAT_REPEAT_SIZE * j], tmpBuffer2[FLOAT_NUM_PER_BLK * j], FLOAT_NUM_PER_BLK);
        }
        ResetMask();
        Adds<float>(tmpBuffer2, inMaxTensor[offset2], 0, tiling.reduceSize);
#elif __CCE_AICORE__ == 220
        Brcb(tmpBuffer1, tmpBuffer2, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        Copy<float, false>(tmpBuffer2, inMaxTensor[offset2], MASK_PLACEHOLDER, reduceCeilValue,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
#endif
        PipeBarrier<PIPE_V>();
        Max<float, false>(maxTensor[offset2], tmpBuffer2, tmpBuffer1, MASK_PLACEHOLDER, reduceCeilValue,
            binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Sub<float, false>(dst[offset1 + FLOAT_REPEAT_SIZE * j], src[offset1 + FLOAT_REPEAT_SIZE * j],
                maxTensor[offset2], MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }

        // expmax = exp(inmax - max)
        Sub<float, false>(tmpBuffer2, tmpBuffer2, maxTensor[offset2], MASK_PLACEHOLDER, reduceCeilValue,
            binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Exp<float, false>(expMaxTensor[offset2], tmpBuffer2, MASK_PLACEHOLDER, reduceCeilValue,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();

        Exp<float, false>(dst[offset1], dst[offset1], MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();

        BasicBlockReduceSumImpl(tmpBuffer3, dst[offset1], tmpBuffer1, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        SetWaitFlagVToS();
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(tmpBuffer1[FLOAT_REPEAT_SIZE * j], tmpBuffer3[FLOAT_NUM_PER_BLK * j], FLOAT_NUM_PER_BLK);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(tmpBuffer1, tmpBuffer3, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
#endif
        PipeBarrier<PIPE_V>();
        // update sum = expmax * insum + sum
        Mul<float, false>(inExpSumTensor[offset2], expMaxTensor[offset2], inExpSumTensor[offset2], MASK_PLACEHOLDER,
            reduceCeilValue, binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Add<float, false>(expSumTensor[offset2], inExpSumTensor[offset2], tmpBuffer1, MASK_PLACEHOLDER, reduceCeilValue,
            binaryRepeatParams);
    }
}

__aicore__ inline void SoftmaxFlashV2BasicBlock(const LocalTensor<half>& dst, const LocalTensor<float>& expSumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor,
    const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitSize];
    // inmax reuse tmpBuffer2
    const LocalTensor<float>& inMaxTmp = workLocal[tiling.splitSize];
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize + tiling.reduceSize];
    // insum reuse tmpbuffer1
    const LocalTensor<float>& inSumTmp =
        workLocal[tiling.splitSize + tiling.reduceSize + tiling.splitM * FLOAT_REPEAT_SIZE / B16_BYTE_SIZE];

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
    const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    BinaryRepeatParams binaryRepeatParams;
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        Cast<float, half, false>(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        BasicBlockReduceMaxImpl(tmpBuffer2, tmpBuffer0, tmpBuffer1, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        SetWaitFlagVToS();
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(tmpBuffer1[FLOAT_REPEAT_SIZE * j], tmpBuffer2[FLOAT_NUM_PER_BLK * j], FLOAT_NUM_PER_BLK);
        }
        ResetMask();
        Adds<float>(inMaxTmp, inMaxTensor[offset2], 0, tiling.reduceSize);
#elif __CCE_AICORE__ == 220
        Brcb(tmpBuffer1, tmpBuffer2, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();

        Copy<float, false>(inMaxTmp, inMaxTensor[offset2], MASK_PLACEHOLDER, reduceCeilValue,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
#endif
        PipeBarrier<PIPE_V>();
        Max<float, false>(maxTensor[offset2], inMaxTmp, tmpBuffer1, MASK_PLACEHOLDER, reduceCeilValue,
            binaryRepeatParams);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < splitBlock; ++i) {
            Sub<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * i], tmpBuffer0[FLOAT_REPEAT_SIZE * i], maxTensor[offset2],
                MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }

        Sub<float, false>(inMaxTmp, inMaxTmp, maxTensor[offset2], MASK_PLACEHOLDER, reduceCeilValue,
            binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Exp<float, false>(inMaxTmp, inMaxTmp, MASK_PLACEHOLDER, reduceCeilValue,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();

        // src 32B copy to dst 64B copy twice
#if __CCE_AICORE__ == 200
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.reduceSize);
        Adds<float, false>(tmpBuffer1, inMaxTmp, 0, MASK_PLACEHOLDER, 1,
            { HALF_FACTOR, 1, DEFAULT_REPEAT_STRIDE * HALF_FACTOR, DEFAULT_REPEAT_STRIDE});
        Adds<float, false>(tmpBuffer1[FLOAT_NUM_PER_BLK], inMaxTmp, 0, MASK_PLACEHOLDER, 1,
            { HALF_FACTOR, 1, DEFAULT_REPEAT_STRIDE * HALF_FACTOR, DEFAULT_REPEAT_STRIDE});
        SetMaskNorm();
        ResetMask();
#elif __CCE_AICORE__ == 220
        Copy<float, false>(tmpBuffer1, inMaxTmp, MASK_PLACEHOLDER, reduceCeilValue,
            { B16_BYTE_SIZE, 1, DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE, DEFAULT_REPEAT_STRIDE });
        Copy<float, false>(tmpBuffer1[DEFAULT_REPEAT_STRIDE], inMaxTmp, MASK_PLACEHOLDER, reduceCeilValue,
            { B16_BYTE_SIZE, 1, DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE, DEFAULT_REPEAT_STRIDE });
#endif
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(expMaxTensor[offset2 * B16_BYTE_SIZE], tmpBuffer1, FLOAT2HALF_ROUND_MODE,
            MASK_PLACEHOLDER, reduceCeilValue * B16_BYTE_SIZE, { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });

        PipeBarrier<PIPE_V>();
        Exp<float, false>(tmpBuffer0, tmpBuffer0, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });

        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        BasicBlockReduceSumImpl(inSumTmp, tmpBuffer0, tmpBuffer1, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();

#if __CCE_AICORE__ == 200
        SetWaitFlagVToS();
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(tmpBuffer1[FLOAT_REPEAT_SIZE * j], inSumTmp[FLOAT_NUM_PER_BLK * j], FLOAT_NUM_PER_BLK);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(tmpBuffer1, inSumTmp, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
#endif
        PipeBarrier<PIPE_V>();

        Mul<float, false>(inSumTmp, inMaxTmp, inExpSumTensor[offset2], MASK_PLACEHOLDER, reduceCeilValue,
            binaryRepeatParams);
        PipeBarrier<PIPE_V>();
        Add<float, false>(expSumTensor[offset2], inSumTmp, tmpBuffer1, MASK_PLACEHOLDER, reduceCeilValue,
            binaryRepeatParams);
    }
}
__aicore__ inline void SoftmaxFlashV2NoUpdateBasicBlock(const LocalTensor<half>& dst,
    const LocalTensor<half>& expSumTensor, const LocalTensor<half>& maxTensor, const LocalTensor<half>& src,
    const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& reduceSumBuffer = workLocal[tiling.splitSize];
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize + tiling.reduceSize];

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
    const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    BinaryRepeatParams binaryRepeatParams;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        SetMaskNorm();
        ResetMask();
        Cast<float, half, false>(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();

        BasicBlockReduceMaxImpl(reduceSumBuffer, tmpBuffer0, tmpBuffer1, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();

#if __CCE_AICORE__ == 200
        SetWaitFlagVToS();
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(tmpBuffer1[FLOAT_REPEAT_SIZE * j * B16_BYTE_SIZE], reduceSumBuffer[FLOAT_NUM_PER_BLK * j],
                FLOAT_NUM_PER_BLK * B16_BYTE_SIZE);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220	
        Brcb(tmpBuffer1, reduceSumBuffer, splitCeilM, { B16_BYTE_SIZE, B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE });
        Brcb(tmpBuffer1[DEFAULT_REPEAT_STRIDE], reduceSumBuffer, splitCeilM,
            { B16_BYTE_SIZE, B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE });
#endif

        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(maxTensor[offset2], tmpBuffer1, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Sub<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer1,
                MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, B16_BYTE_SIZE });
        }
        PipeBarrier<PIPE_V>();
        Exp<float, false>(tmpBuffer0, tmpBuffer0, MASK_PLACEHOLDER, (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE),
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });

        BasicBlockReduceSumImpl(reduceSumBuffer, tmpBuffer0, tmpBuffer1, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();

#if __CCE_AICORE__ == 200
        SetWaitFlagVToS();
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(tmpBuffer1[FLOAT_REPEAT_SIZE * j * B16_BYTE_SIZE], reduceSumBuffer[FLOAT_NUM_PER_BLK * j],
                FLOAT_NUM_PER_BLK * B16_BYTE_SIZE);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(tmpBuffer1, reduceSumBuffer, splitCeilM, { B16_BYTE_SIZE, B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE });
        Brcb(tmpBuffer1[DEFAULT_REPEAT_STRIDE], reduceSumBuffer, splitCeilM,
            { B16_BYTE_SIZE, B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE });
#endif

        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(expSumTensor[offset2], tmpBuffer1, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
}

__aicore__ inline void SoftmaxFlashV2NoUpdateBasicBlock(const LocalTensor<half>& dst,
    const LocalTensor<float>& expSumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& src,
    const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitSize];
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize + tiling.reduceSize];

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
    const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    BinaryRepeatParams binaryRepeatParams;
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        Cast<float, half, false>(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        BasicBlockReduceMaxImpl(tmpBuffer2, tmpBuffer0, tmpBuffer1, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();

#if __CCE_AICORE__ == 200
        SetWaitFlagVToS();
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(maxTensor[offset2 + FLOAT_REPEAT_SIZE * j], tmpBuffer2[FLOAT_NUM_PER_BLK * j],
                FLOAT_NUM_PER_BLK);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(maxTensor[offset2], tmpBuffer2, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
#endif
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < splitBlock; ++i) {
            Sub<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * i], tmpBuffer0[FLOAT_REPEAT_SIZE * i], maxTensor[offset2],
                MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }

        PipeBarrier<PIPE_V>();
        Exp<float, false>(tmpBuffer0, tmpBuffer0, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });

        BasicBlockReduceSumImpl(tmpBuffer2, tmpBuffer0, tmpBuffer1, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();

#if __CCE_AICORE__ == 200
        SetWaitFlagVToS();
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(expSumTensor[offset2 + FLOAT_REPEAT_SIZE * j], tmpBuffer2[FLOAT_NUM_PER_BLK * j],
                FLOAT_NUM_PER_BLK);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(expSumTensor[offset2], tmpBuffer2, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
#endif
    }
}

__aicore__ inline void SoftmaxFlashV2NoUpdateBasicBlock(const LocalTensor<float>& dst,
    const LocalTensor<float>& expSumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<float>& src,
    const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling)
{
    const LocalTensor<float>& tmpBuffer1 = workLocal;
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitM * FLOAT_REPEAT_SIZE];

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
    const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    BinaryRepeatParams binaryRepeatParams;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        PipeBarrier<PIPE_V>();
        BasicBlockReduceMaxImpl(tmpBuffer2, src[offset1], tmpBuffer1, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();

#if __CCE_AICORE__ == 200
        SetWaitFlagVToS();
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(maxTensor[offset2 + FLOAT_REPEAT_SIZE * j], tmpBuffer2[FLOAT_NUM_PER_BLK * j],
                FLOAT_NUM_PER_BLK);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(maxTensor[offset2], tmpBuffer2, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
#endif

        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Sub<float, false>(src[offset1 + FLOAT_REPEAT_SIZE * j], src[offset1 + FLOAT_REPEAT_SIZE * j],
                maxTensor[offset2], MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();
        Exp<float, false>(dst[offset1], src[offset1], MASK_PLACEHOLDER, (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE),
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();

        BasicBlockReduceSumImpl(tmpBuffer2, dst[offset1], tmpBuffer1, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        SetWaitFlagVToS();
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(expSumTensor[offset2 + FLOAT_REPEAT_SIZE * j], tmpBuffer2[FLOAT_NUM_PER_BLK * j],
                FLOAT_NUM_PER_BLK);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(expSumTensor[offset2], tmpBuffer2, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
#endif
    }
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_BASIC_BLOCK_IMPL_H