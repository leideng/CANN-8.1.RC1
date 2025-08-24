/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_flashv2_update_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_UPDATE_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_UPDATE_IMPL_H


namespace AscendC {

__aicore__ inline void SoftmaxFlashV2UpdateImpl(const LocalTensor<half>& dst, const LocalTensor<half>& expSumTensor,
    const LocalTensor<half>& maxTensor, const LocalTensor<half>& src, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<half>& inExpSumTensor, const LocalTensor<half>& inMaxTensor, const LocalTensor<float>& workLocal,
    const ReduceLastND& reduceParam, const SoftMaxTiling& tiling, const uint32_t& offset1, const uint32_t& offset2,
    const uint32_t& splitSize, const uint32_t& reduceSize)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize];
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitSize + tiling.reduceSize]; // need splitM * 64
    const LocalTensor<float>& tmpBuffer3 =
        workLocal[tiling.splitSize + tiling.reduceSize + tiling.splitM * FLOAT_REPEAT_SIZE];

    Cast(tmpBuffer1, inMaxTensor[offset2], RoundMode::CAST_NONE, reduceSize);
    Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
    PipeBarrier<PIPE_V>();
    NewReduceMaxLastNDImpl(tmpBuffer3, tmpBuffer0, tmpBuffer2, reduceParam);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceSize);

    Max<float, false>(tmpBuffer3, tmpBuffer1, tmpBuffer3, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    Sub<float, false>(tmpBuffer1, tmpBuffer1, tmpBuffer3, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    Cast<half, float, false>(maxTensor[offset2], tmpBuffer3, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    SetMaskNorm();
    ResetMask();

    GenericSubNDImpl(tmpBuffer0, tmpBuffer0, tmpBuffer3, reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitSize);

    Exp<float, false>(tmpBuffer0, tmpBuffer0, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Cast<half, float, false>(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    SetMaskNorm();
    ResetMask();

    NewReduceSumLastNDImpl(tmpBuffer3, tmpBuffer0, tmpBuffer2, reduceParam);

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceSize);
    Exp<float, false>(tmpBuffer1, tmpBuffer1, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Cast<half, float, false>(expMaxTensor[offset2], tmpBuffer1, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    Cast<float, half, false>(tmpBuffer2, inExpSumTensor[offset2], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Mul<float, false>(tmpBuffer1, tmpBuffer1, tmpBuffer2, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Add<float, false>(tmpBuffer2, tmpBuffer1, tmpBuffer3, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Cast<half, float, false>(expSumTensor[offset2], tmpBuffer2, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void SoftmaxFlashV2UpdateImpl(const LocalTensor<float>& dst, const LocalTensor<float>& expSumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<float>& src, const LocalTensor<float>& expMaxTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor,
    const LocalTensor<float>& workLocal, const ReduceLastND& reduceParam, const SoftMaxTiling& tiling,
    const uint32_t& offset1, const uint32_t& offset2, const uint32_t& splitSize, const uint32_t& reduceSize)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.reduceSize]; // need splitM * 64

    NewReduceMaxLastNDImpl(tmpBuffer0, src[offset1], tmpBuffer1, reduceParam);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceSize);

    Max<float, false>(tmpBuffer0, inMaxTensor[offset2], tmpBuffer0, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    Sub<float, false>(tmpBuffer1, inMaxTensor[offset2], tmpBuffer0, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    Exp<float, false>(expMaxTensor[offset2], tmpBuffer1, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    SetMaskNorm();
    ResetMask();
    GenericSubNDImpl(dst[offset1], src[offset1], tmpBuffer0, reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);

    PipeBarrier<PIPE_V>();
    Exp(dst[offset1], dst[offset1], splitSize); // exp(x - max)
#if __CCE_AICORE__ == 200
    Adds<float>(maxTensor[offset2], tmpBuffer0, 0, reduceSize);
#elif __CCE_AICORE__ == 220
    DataCopy(maxTensor[offset2], tmpBuffer0, reduceSize);
#endif
    PipeBarrier<PIPE_V>();
    NewReduceSumLastNDImpl(tmpBuffer0, dst[offset1], tmpBuffer1, reduceParam);

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceSize);

    Mul<float, false>(expSumTensor[offset2], expMaxTensor[offset2], inExpSumTensor[offset2], MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Add<float, false>(expSumTensor[offset2], expSumTensor[offset2], tmpBuffer0, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void SoftmaxFlashV2UpdateImpl(const LocalTensor<half>& dst, const LocalTensor<float>& expSumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor,
    const LocalTensor<float>& workLocal, const ReduceLastND& reduceParam, const SoftMaxTiling& tiling,
    const uint32_t& offset1, const uint32_t& offset2, const uint32_t& splitSize, const uint32_t& reduceSize)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize];
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitSize + tiling.reduceSize];
    const LocalTensor<float>& tmpBuffer3 =
        workLocal[tiling.splitSize + tiling.reduceSize + tiling.splitM * FLOAT_REPEAT_SIZE];
#if __CCE_AICORE__ == 200
    Adds<float>(tmpBuffer1, inMaxTensor[offset2], 0, reduceSize);
#elif __CCE_AICORE__ == 220
    DataCopy(tmpBuffer1, inMaxTensor[offset2], reduceSize);
#endif
    Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
    PipeBarrier<PIPE_V>();
    NewReduceMaxLastNDImpl(tmpBuffer3, tmpBuffer0, tmpBuffer2, reduceParam);
    PipeBarrier<PIPE_V>();
    Max(maxTensor[offset2], inMaxTensor[offset2], tmpBuffer3, reduceSize);
    PipeBarrier<PIPE_V>();
    GenericSubNDImpl(tmpBuffer0, tmpBuffer0, maxTensor[offset2], reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);
    PipeBarrier<PIPE_V>();
    Exp(tmpBuffer0, tmpBuffer0, splitSize);
    PipeBarrier<PIPE_V>();
    Cast(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, splitSize);

    NewReduceSumLastNDImpl(tmpBuffer3, tmpBuffer0, tmpBuffer2, reduceParam);

    Sub(tmpBuffer1, tmpBuffer1, maxTensor[offset2], reduceSize);
    PipeBarrier<PIPE_V>();
    Exp(tmpBuffer1, tmpBuffer1, reduceSize);
    PipeBarrier<PIPE_V>();

    BroadCastLastImpl(tmpBuffer0, tmpBuffer1,
        { tiling.reduceM, B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE, tiling.reduceM, tiling.reduceK });
    PipeBarrier<PIPE_V>();
    Cast(expMaxTensor[offset2 * B16_BYTE_SIZE], tmpBuffer0, FLOAT2HALF_ROUND_MODE, reduceSize * B16_BYTE_SIZE);

    Mul(tmpBuffer1, tmpBuffer1, inExpSumTensor[offset2], reduceSize);
    PipeBarrier<PIPE_V>();
    Add(expSumTensor[offset2], tmpBuffer1, tmpBuffer3, reduceSize);
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_UPDATE_IMPL_H