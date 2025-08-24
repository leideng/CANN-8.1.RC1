/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_basic_block_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_BASIC_BLOCK_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_BASIC_BLOCK_IMPL_H

namespace AscendC {

__aicore__ inline void SoftMaxBasicBlock(const LocalTensor<half>& dst, const LocalTensor<half>& sumTensor,
    const LocalTensor<half>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize];
    const LocalTensor<float>& reduceSumBuffer = workLocal[tiling.splitSize + tiling.splitM * FLOAT_REPEAT_SIZE];

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
    const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    const uint32_t halfSplitSize = tiling.splitSize / HALF_FACTOR;
    BinaryRepeatParams binaryRepeatParams;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        SetMaskNorm();
        ResetMask();
        Cast<float, half, false>(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_BLK_NUM, HALF_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();

        if (splitBlock == 1) {
            BlockReduceMax<float, false>(tmpBuffer1, tmpBuffer0, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        } else {
            BasicBlockMaxImpl(tmpBuffer1, tmpBuffer0, (uint8_t)(tiling.splitM), offset, splitBlock);
            PipeBarrier<PIPE_V>();
            BlockReduceMax<float, false>(tmpBuffer1, tmpBuffer1, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                DEFAULT_BLK_NUM);
        }

        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(reduceSumBuffer, tmpBuffer1, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_BLK_NUM);
        PipeBarrier<PIPE_V>();

#if __CCE_AICORE__ == 200
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(tmpBuffer1[FLOAT_REPEAT_SIZE * j * HALF_FACTOR], reduceSumBuffer[FLOAT_NUM_PER_BLK * j],
                HALF_FACTOR * DEFAULT_REPEAT_STRIDE);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(tmpBuffer1, reduceSumBuffer, splitCeilM, { HALF_FACTOR, HALF_FACTOR * DEFAULT_REPEAT_STRIDE });
        Brcb(tmpBuffer1[DEFAULT_BLK_NUM], reduceSumBuffer, splitCeilM,
            { HALF_FACTOR, HALF_FACTOR * DEFAULT_REPEAT_STRIDE });

#endif
        PipeBarrier<PIPE_V>();

        Cast<half, float, false>(maxTensor[offset2], tmpBuffer1, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });

        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Sub<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer1,
                MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, HALF_FACTOR });
        }
        PipeBarrier<PIPE_V>();
        Exp<float, false>(tmpBuffer0, tmpBuffer0, MASK_PLACEHOLDER, (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE),
            { 1, 1, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM });
        PipeBarrier<PIPE_V>();

        if (splitBlock == 1) {
            BlockReduceSum<float, false>(tmpBuffer1, tmpBuffer0, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                DEFAULT_BLK_NUM);
        } else {
            BasicBlockAddImpl(tmpBuffer1, tmpBuffer0, (uint8_t)(tiling.splitM), offset, splitBlock);
            PipeBarrier<PIPE_V>();
            BlockReduceSum<float, false>(tmpBuffer1, tmpBuffer1, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                DEFAULT_BLK_NUM);
        }

        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(reduceSumBuffer, tmpBuffer1, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_BLK_NUM);
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(tmpBuffer1[FLOAT_REPEAT_SIZE * j * HALF_FACTOR], reduceSumBuffer[FLOAT_NUM_PER_BLK * j],
                HALF_FACTOR * DEFAULT_REPEAT_STRIDE);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(tmpBuffer1, reduceSumBuffer, splitCeilM, { HALF_FACTOR, HALF_FACTOR * DEFAULT_REPEAT_STRIDE });
        Brcb(tmpBuffer1[DEFAULT_BLK_NUM], reduceSumBuffer, splitCeilM,
            { HALF_FACTOR, HALF_FACTOR * DEFAULT_REPEAT_STRIDE });

#endif
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(sumTensor[offset2], tmpBuffer1, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });

        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Div<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer1,
                MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, HALF_FACTOR });
        }
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_BLK_NUM });
    }
}

__aicore__ inline void SoftMaxBasicBlock(const LocalTensor<float>& dst, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<float>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling)
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
    const uint32_t halfSplitSize = tiling.splitSize / HALF_FACTOR;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        SetMaskNorm();
        ResetMask();

        if (splitBlock == 1) {
            BlockReduceMax<float, false>(tmpBuffer1, src[offset1], (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        } else {
            BasicBlockMaxImpl(tmpBuffer1, src[offset1], (uint8_t)(tiling.splitM), offset, splitBlock);
            PipeBarrier<PIPE_V>();
            BlockReduceMax<float, false>(tmpBuffer1, tmpBuffer1, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        }

        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(tmpBuffer2, tmpBuffer1, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
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
            Sub<float, false>(dst[offset1 + FLOAT_REPEAT_SIZE * j], src[offset1 + FLOAT_REPEAT_SIZE * j],
                maxTensor[offset2], MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();
        Exp<float, false>(dst[offset1], dst[offset1], MASK_PLACEHOLDER, (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE),
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();

        if (splitBlock == 1) {
            BlockReduceSum<float, false>(tmpBuffer1, dst[offset1], (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        } else {
            BasicBlockAddImpl(tmpBuffer1, dst[offset1], (uint8_t)(tiling.splitM), offset, splitBlock);
            PipeBarrier<PIPE_V>();
            BlockReduceSum<float, false>(tmpBuffer1, tmpBuffer1, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        }

        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(tmpBuffer2, tmpBuffer1, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(sumTensor[offset2 + FLOAT_REPEAT_SIZE * j], tmpBuffer2[FLOAT_NUM_PER_BLK * j],
                FLOAT_NUM_PER_BLK);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(sumTensor[offset2], tmpBuffer2, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
#endif
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Div<float, false>(dst[offset1 + FLOAT_REPEAT_SIZE * j], dst[offset1 + FLOAT_REPEAT_SIZE * j],
                sumTensor[offset2], MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void SoftMaxBasicBlock(const LocalTensor<half>& dst, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize];
    const LocalTensor<float>& reduceSumBuffer = workLocal[tiling.splitSize + tiling.splitM * FLOAT_REPEAT_SIZE];

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    uint8_t stride = (uint8_t)(tiling.splitK / FLOAT_NUM_PER_BLK);
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        SetMaskNorm();
        ResetMask();
        Cast<float, half, false>(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();

        if (splitBlock == 1) {
            BlockReduceMax<float, false>(tmpBuffer1, tmpBuffer0, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        } else {
            BasicBlockMaxImpl(tmpBuffer1, tmpBuffer0, (uint8_t)(tiling.splitM), offset, splitBlock);
            PipeBarrier<PIPE_V>();
            BlockReduceMax<float, false>(tmpBuffer1, tmpBuffer1, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        }
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(reduceSumBuffer, tmpBuffer1, splitCeilM, MASK_PLACEHOLDER, 1, 1,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(maxTensor[offset2 + FLOAT_REPEAT_SIZE * j], reduceSumBuffer[FLOAT_NUM_PER_BLK * j],
                FLOAT_NUM_PER_BLK);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(maxTensor[offset2], reduceSumBuffer, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
#endif
        PipeBarrier<PIPE_V>();

        for (uint32_t j = 0; j < splitBlock; ++j) {
            Sub<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer0[FLOAT_REPEAT_SIZE * j], maxTensor[offset2],
                MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();
        Exp<float, false>(tmpBuffer0, tmpBuffer0, MASK_PLACEHOLDER, (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE),
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();

        if (splitBlock == 1) {
            BlockReduceSum<float, false>(tmpBuffer1, tmpBuffer0, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        } else {
            BasicBlockAddImpl(tmpBuffer1, tmpBuffer0, (uint8_t)(tiling.splitM), offset, splitBlock);
            PipeBarrier<PIPE_V>();
            BlockReduceSum<float, false>(tmpBuffer1, tmpBuffer1, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                DEFAULT_REPEAT_STRIDE);
        }

        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(reduceSumBuffer, tmpBuffer1, splitCeilM, MASK_PLACEHOLDER, 1, 1,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (uint32_t j = 0; j < splitCeilM; j++) {
            AlignedBrcbImpl(sumTensor[offset2 + FLOAT_REPEAT_SIZE * j], reduceSumBuffer[FLOAT_NUM_PER_BLK * j],
                FLOAT_NUM_PER_BLK);
        }
        ResetMask();
#elif __CCE_AICORE__ == 220
        Brcb(sumTensor[offset2], reduceSumBuffer, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
#endif
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Div<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer0[FLOAT_REPEAT_SIZE * j], sumTensor[offset2],
                MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
}

}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_BASIC_BLOCK_IMPL_H