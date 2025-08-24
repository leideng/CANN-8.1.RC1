/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_grad_nd_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_COMMON_SOFTMAX_GRAD_ND_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_COMMON_SOFTMAX_GRAD_ND_IMPL_H

namespace AscendC {

__aicore__ inline void SoftmaxGradNDGenericImpl(const LocalTensor<half>& dstTensor, const LocalTensor<half>& gradTensor,
    const LocalTensor<half>& srcTensor, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling,
    const ReduceLastND& reduceSumParam, const BroadCastLastND& brcParam, const bool isFront, const uint32_t offset1,
    const uint32_t offset2, const uint32_t splitSize, const uint32_t reduceSize)
{
    LocalTensor<float> srcBuffer = workLocal;
    LocalTensor<float> gradBuffer = workLocal[tiling.splitSize];
    LocalTensor<float> dstBuffer = workLocal[tiling.splitSize + tiling.splitSize];

    LocalTensor<float> reduceBuffer = workLocal[tiling.splitSize + tiling.splitSize + tiling.splitSize];
    LocalTensor<float> addBuffer =
        workLocal[tiling.splitSize + tiling.splitSize + tiling.splitSize + tiling.reduceSize];

    Cast(srcBuffer, srcTensor[offset1], RoundMode::CAST_NONE, splitSize);
    Cast(gradBuffer, gradTensor[offset1], RoundMode::CAST_NONE, splitSize);
    PipeBarrier<PIPE_V>();
    Mul(dstBuffer, srcBuffer, gradBuffer, splitSize);
    PipeBarrier<PIPE_V>();
    ReduceSumLastNDImpl(addBuffer, dstBuffer, reduceBuffer, reduceSumParam);
    PipeBarrier<PIPE_V>();
    if (isFront) {
        Cast(dstTensor[offset2], addBuffer, FLOAT2HALF_ROUND_MODE, reduceSize);
    } else {
        BroadCastLastND brcParam = { tiling.reduceM, tiling.srcK, tiling.reduceM, tiling.reduceK };
        BroadCastLastImpl(dstBuffer, addBuffer, brcParam);
        PipeBarrier<PIPE_V>();
        Sub(dstBuffer, gradBuffer, dstBuffer, splitSize);
        PipeBarrier<PIPE_V>();
        Mul(dstBuffer, dstBuffer, srcBuffer, splitSize);
        PipeBarrier<PIPE_V>();
        Cast(dstTensor[offset1], dstBuffer, FLOAT2HALF_ROUND_MODE, splitSize);
    }
}

__aicore__ inline void SoftmaxGradNDImpl(const LocalTensor<half>& dstTensor, const LocalTensor<half>& gradTensor,
    const LocalTensor<half>& srcTensor, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling,
    const LastAxisShapeND& originalSrcShape, bool isFront = false)
{
    const ReduceLastND reduceSumParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
                                          tiling.splitK, tiling.reduceM, tiling.reduceK };
    const BroadCastLastND brcParam = { tiling.reduceM, tiling.srcK, tiling.reduceM, tiling.reduceK };
    LocalTensor<float> srcBuffer = workLocal;
    LocalTensor<float> gradBuffer = workLocal[tiling.splitSize];
    LocalTensor<float> dstBuffer = workLocal[tiling.splitSize + tiling.splitSize];

    LocalTensor<float> reduceBuffer = workLocal[tiling.splitSize + tiling.splitSize + tiling.splitSize];
    LocalTensor<float> addBuffer =
        workLocal[tiling.splitSize + tiling.splitSize + tiling.splitSize + tiling.reduceSize];

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        SoftmaxGradNDGenericImpl(dstTensor, gradTensor, srcTensor, workLocal, tiling, reduceSumParam, brcParam, isFront,
            offset1, offset2, tiling.splitSize, tiling.reduceSize);
        offset1 += tiling.splitSize;
        offset2 += tiling.reduceSize;
    }
    if (tiling.tailM != 0) {
        const ReduceLastND tailReduceSumParam = { tiling.tailM,  originalSrcShape.k, tiling.tailM,
                                                  tiling.splitK, tiling.tailM, tiling.reduceK };
        const BroadCastLastND tailBrcParam = { tiling.tailM, tiling.srcK, tiling.tailM, tiling.reduceK };
        SoftmaxGradNDGenericImpl(dstTensor, gradTensor, srcTensor, workLocal, tiling, tailReduceSumParam, tailBrcParam,
            isFront, offset1, offset2, tiling.tailSplitSize, tiling.tailReduceSize);
    }
}

template <typename T, bool isBasicBlock = false>
__aicore__ inline void SoftmaxGradFrontNDImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling,
    const LastAxisShapeND& originalSrcShape)
{
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
    ReduceLastND reduceSumParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
        tiling.splitK, tiling.reduceM,     tiling.reduceK };

    if constexpr (sizeof(T) == sizeof(half)) {
        LocalTensor<float> srcBuffer = workLocal;
        LocalTensor<float> gradBuffer = workLocal[tiling.splitSize];
        LocalTensor<float> dstBuffer = workLocal[tiling.splitSize + tiling.splitSize];

        LocalTensor<float> reduceBuffer = workLocal[tiling.splitSize + tiling.splitSize + tiling.splitSize];
        LocalTensor<float> addBuffer =
            workLocal[tiling.splitSize + tiling.splitSize + tiling.splitSize + tiling.reduceSize];
        const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
        const uint32_t elementNumPerBlk = DEFAULT_C0_SIZE / B32_BYTE_SIZE;
        uint8_t offset = (uint8_t)(splitBlock * elementNumPerBlk);
        const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
        const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
        const uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
        SetMaskNorm();
        ResetMask();
        for (uint32_t i = 0; i < tiling.rangeM; i++) {
            if constexpr (isBasicBlock) {
                Cast<float, half, false>(srcBuffer, srcTensor[i * tiling.splitSize], RoundMode::CAST_NONE,
                    MASK_PLACEHOLDER, repeatTimes, { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
                Cast<float, half, false>(gradBuffer, gradTensor[i * tiling.splitSize], RoundMode::CAST_NONE,
                    MASK_PLACEHOLDER, repeatTimes, { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
                PipeBarrier<PIPE_V>();
                Mul<float, false>(dstBuffer, srcBuffer, gradBuffer, MASK_PLACEHOLDER, repeatTimes,
                    { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
                for (uint32_t j = 1; j < splitBlock; ++j) {
                    PipeBarrier<PIPE_V>();
                    Add<float, false>(dstBuffer, dstBuffer, dstBuffer[FLOAT_REPEAT_SIZE * j], MASK_PLACEHOLDER,
                        (uint8_t)(tiling.splitM), { 1, 1, 1, offset, offset, offset });
                }
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float, false>(dstBuffer, dstBuffer, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                    offset);
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float, false>(reduceBuffer, dstBuffer, splitCeilM, MASK_PLACEHOLDER, 1, 1,
                    DEFAULT_REPEAT_STRIDE);
                PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
                event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventIdVToS);
                WaitFlag<HardEvent::V_S>(eventIdVToS);
                for (uint32_t j = 0; j < splitCeilM; j++) {
                    AlignedBrcbImpl(dstBuffer[FLOAT_REPEAT_SIZE * j * B16_BYTE_SIZE],
                        reduceBuffer[FLOAT_NUM_PER_BLK * j], DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE);
                }
                ResetMask();
#elif __CCE_AICORE__ == 220
                Brcb(dstBuffer, reduceBuffer, splitCeilM, { B16_BYTE_SIZE, DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE });
                Brcb(dstBuffer[DEFAULT_BLK_NUM], reduceBuffer, splitCeilM,
                    { B16_BYTE_SIZE, DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE });
#endif
                PipeBarrier<PIPE_V>();
                Cast<half, float, false>(dstTensor[i * tiling.reduceSize], dstBuffer, FLOAT2HALF_ROUND_MODE,
                    MASK_PLACEHOLDER, reduceCeilValue, { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
            } else {
                Cast(srcBuffer, srcTensor[i * tiling.splitSize], RoundMode::CAST_NONE, tiling.splitSize);
                Cast(gradBuffer, gradTensor[i * tiling.splitSize], RoundMode::CAST_NONE, tiling.splitSize);
                PipeBarrier<PIPE_V>();
                Mul(dstBuffer, srcBuffer, gradBuffer, tiling.splitSize);
                PipeBarrier<PIPE_V>();
                ReduceSumLastNDImpl(addBuffer, dstBuffer, reduceBuffer, reduceSumParam);
                PipeBarrier<PIPE_V>();
                Cast(dstTensor[i * tiling.reduceSize], addBuffer, FLOAT2HALF_ROUND_MODE, tiling.reduceSize);
            }
        }
        if (tiling.tailM != 0) {
            Cast(srcBuffer, srcTensor[tiling.rangeM * tiling.splitSize], RoundMode::CAST_NONE, tiling.tailSplitSize);
            Cast(gradBuffer, gradTensor[tiling.rangeM * tiling.splitSize], RoundMode::CAST_NONE, tiling.tailSplitSize);
            PipeBarrier<PIPE_V>();
            Mul(dstBuffer, srcBuffer, gradBuffer, tiling.tailSplitSize);
            reduceSumParam.srcM = tiling.tailM;
            reduceSumParam.dstM = tiling.tailM;
            reduceSumParam.originalSrcM = tiling.tailM;
            PipeBarrier<PIPE_V>();
            ReduceSumLastNDImpl(addBuffer, dstBuffer, reduceBuffer, reduceSumParam);
            PipeBarrier<PIPE_V>();
            Cast(dstTensor[tiling.rangeM * tiling.reduceSize], addBuffer, FLOAT2HALF_ROUND_MODE, tiling.tailReduceSize);
        }
    } else {
        LocalTensor<float> srcBuffer = workLocal;
        LocalTensor<float> reduceBuffer = workLocal[tiling.splitSize];
        uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
        uint32_t offset1 = 0;
        uint32_t offset2 = 0;
        const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
        const uint32_t elementNumPerBlk = DEFAULT_C0_SIZE / B32_BYTE_SIZE;
        uint8_t offset = (uint8_t)(splitBlock * elementNumPerBlk);
        const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, elementNumPerBlk));
        SetMaskNorm();
        ResetMask();
        for (uint32_t i = 0; i < tiling.rangeM; i++) {
            if constexpr (isBasicBlock) {
                offset2 = i * tiling.reduceSize;
                offset1 = i * tiling.splitSize;
                PipeBarrier<PIPE_V>();
                Mul<float, false>(srcBuffer, srcTensor[offset1], gradTensor[offset1], MASK_PLACEHOLDER, repeatTimes,
                    { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });

                for (uint32_t j = 1; j < splitBlock; ++j) {
                    PipeBarrier<PIPE_V>();
                    Add<float, false>(srcBuffer, srcBuffer, srcBuffer[FLOAT_REPEAT_SIZE * j], MASK_PLACEHOLDER,
                        (uint8_t)(tiling.splitM), { 1, 1, 1, offset, offset, offset });
                }
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float, false>(srcBuffer, srcBuffer, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                    splitBlock * DEFAULT_REPEAT_STRIDE);
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float, false>(reduceBuffer, srcBuffer, splitCeilM, MASK_PLACEHOLDER, 1, 1,
                    DEFAULT_REPEAT_STRIDE);
                PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
                event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventIdVToS);
                WaitFlag<HardEvent::V_S>(eventIdVToS);
                for (uint32_t j = 0; j < splitCeilM; j++) {
                    AlignedBrcbImpl(dstTensor[offset2 + FLOAT_REPEAT_SIZE * j], reduceBuffer[FLOAT_NUM_PER_BLK * j],
                        FLOAT_NUM_PER_BLK);
                }
                ResetMask();
#elif __CCE_AICORE__ == 220
                Brcb(dstTensor[offset2], reduceBuffer, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE });
#endif
            } else {
                Mul(srcBuffer, srcTensor[i * tiling.splitSize], gradTensor[i * tiling.splitSize], tiling.splitSize);
                PipeBarrier<PIPE_V>();
                ReduceSumLastNDImpl(dstTensor[i * tiling.reduceSize], srcBuffer, reduceBuffer, reduceSumParam);
                PipeBarrier<PIPE_V>();
            }
        }

        if (tiling.tailM != 0) {
            Mul(srcBuffer, srcTensor[tiling.rangeM * tiling.splitSize], gradTensor[tiling.rangeM * tiling.splitSize],
                tiling.tailSplitSize);
            PipeBarrier<PIPE_V>();

            reduceSumParam.srcM = tiling.tailM;
            reduceSumParam.dstM = tiling.tailM;
            reduceSumParam.originalSrcM = tiling.tailM;
            ReduceSumLastNDImpl(dstTensor[tiling.rangeM * tiling.reduceSize], srcBuffer, reduceBuffer, reduceSumParam);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline void SoftmaxGradPostProcess(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling,
    const LastAxisShapeND& originalSrcShape, bool isFront = false)
{
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
    ReduceLastND reduceSumParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
        tiling.splitK, tiling.reduceM,     tiling.reduceK };

    if constexpr (sizeof(T) == sizeof(half)) {
        SoftmaxGradNDImpl(dstTensor, gradTensor, srcTensor, workLocal, tiling, originalSrcShape, isFront);
    } else {
        if (isFront) {
            SoftmaxGradFrontNDImpl<float>(dstTensor, srcTensor, gradTensor, workLocal, tiling, originalSrcShape);
        } else {
            LocalTensor<float> splitBuffer = workLocal;
            LocalTensor<float> reduceBuffer = workLocal[tiling.splitSize];
            LocalTensor<float> addBuffer = workLocal[tiling.splitSize + tiling.reduceSize];

            BroadCastLastND brcParam = { tiling.reduceM, tiling.srcK, tiling.reduceM, elementNumPerBlk };
            for (uint32_t i = 0; i < tiling.rangeM; i++) {
                Mul(splitBuffer, srcTensor[i * tiling.splitSize], gradTensor[i * tiling.splitSize], tiling.splitSize);
                PipeBarrier<PIPE_V>();
                ReduceSumLastNDImpl(addBuffer, splitBuffer, reduceBuffer, reduceSumParam);
                PipeBarrier<PIPE_V>();
                BroadCastLastImpl(splitBuffer, addBuffer, brcParam);
                PipeBarrier<PIPE_V>();
                Sub(splitBuffer, gradTensor[i * tiling.splitSize], splitBuffer, tiling.splitSize);
                PipeBarrier<PIPE_V>();
                Mul(dstTensor[i * tiling.splitSize], srcTensor[i * tiling.splitSize], splitBuffer, tiling.splitSize);
            }
            if (tiling.tailM != 0) {
                reduceSumParam.srcM = tiling.tailM;
                reduceSumParam.dstM = tiling.tailM;
                reduceSumParam.originalSrcM = tiling.tailM;
                Mul(splitBuffer, srcTensor[tiling.rangeM * tiling.splitSize],
                    gradTensor[tiling.rangeM * tiling.splitSize], tiling.tailSplitSize);
                PipeBarrier<PIPE_V>();
                ReduceSumLastNDImpl(addBuffer, splitBuffer, reduceBuffer, reduceSumParam);
                PipeBarrier<PIPE_V>();
                BroadCastLastImpl(splitBuffer, addBuffer, brcParam);
                PipeBarrier<PIPE_V>();
                Sub(splitBuffer, gradTensor[tiling.rangeM * tiling.splitSize], splitBuffer, tiling.tailSplitSize);
                PipeBarrier<PIPE_V>();
                Mul(dstTensor[tiling.rangeM * tiling.splitSize], srcTensor[tiling.rangeM * tiling.splitSize],
                    splitBuffer, tiling.tailSplitSize);
            }
        }
    }
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_COMMON_SOFTMAX_GRAD_ND_IMPL_H