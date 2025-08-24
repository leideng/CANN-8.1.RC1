/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_flashv2_no_update_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_NO_UPDATE_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_NO_UPDATE_IMPL_H


namespace AscendC {
__aicore__ inline void SoftmaxFlashV2NoUpdateImpl(const LocalTensor<half>& dst, const LocalTensor<half>& expSumTensor,
    const LocalTensor<half>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const ReduceLastND& reduceParam, const SoftMaxTiling& tiling, const uint32_t& offset1, const uint32_t& offset2,
    const uint32_t& splitSize, const uint32_t& reduceSize)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& reduceBuffer = workLocal[tiling.splitSize];
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitSize + tiling.reduceSize]; // need splitM * 64

    Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
    PipeBarrier<PIPE_V>();
    NewReduceMaxLastNDImpl(reduceBuffer, tmpBuffer0, tmpBuffer2, reduceParam);
    PipeBarrier<PIPE_V>();
    GenericSubNDImpl(tmpBuffer0, tmpBuffer0, reduceBuffer, reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);
    PipeBarrier<PIPE_V>();
    Cast(maxTensor[offset2], reduceBuffer, FLOAT2HALF_ROUND_MODE, reduceSize);
    Exp(tmpBuffer0, tmpBuffer0, splitSize);
    PipeBarrier<PIPE_V>();
    Cast(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, splitSize);
    PipeBarrier<PIPE_V>();
    NewReduceSumLastNDImpl(reduceBuffer, tmpBuffer0, tmpBuffer2, reduceParam);
    PipeBarrier<PIPE_V>();
    Cast(expSumTensor[offset2], reduceBuffer, FLOAT2HALF_ROUND_MODE, reduceSize);
}

__aicore__ inline void SoftmaxFlashV2NoUpdateImpl(const LocalTensor<float>& dst, const LocalTensor<float>& expSumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<float>& src, const LocalTensor<float>& workLocal,
    const ReduceLastND& reduceParam, const SoftMaxTiling& tiling, const uint32_t& offset1, const uint32_t& offset2,
    const uint32_t& splitSize, const uint32_t& reduceSize)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;

    NewReduceMaxLastNDImpl(maxTensor[offset2], src[offset1], tmpBuffer0, reduceParam);
    PipeBarrier<PIPE_V>();
    GenericSubNDImpl(dst[offset1], src[offset1], maxTensor[offset2], reduceParam.originalSrcM, tiling.srcK,
        tiling.reduceK);
    PipeBarrier<PIPE_V>();
    Exp(dst[offset1], dst[offset1], splitSize);
    PipeBarrier<PIPE_V>();
    NewReduceSumLastNDImpl(expSumTensor[offset2], dst[offset1], tmpBuffer0, reduceParam);
}

template <typename T>
__aicore__ inline void SoftmaxFlashV2NoUpdateExtImpl(const LocalTensor<T>& dst, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling, ReduceLastND& reduceParam)
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitSize = tiling.splitSize;
    uint32_t reduceSize = tiling.reduceSize;
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i <= tiling.rangeM; i++) {
        SoftmaxFlashV2NoUpdateImpl(dst, expSumTensor, maxTensor, src, workLocal, reduceParam, tiling, offset1,
            offset2, splitSize, reduceSize);
        offset1 += tiling.splitSize;
        offset2 += tiling.reduceSize;
        if (i == (tiling.rangeM - 1)) {
            if (tiling.tailM == 0) {
                break;
            }
            offset2 = tiling.rangeM * tiling.reduceSize;
            offset1 = tiling.rangeM * tiling.splitSize;
            splitSize = tiling.tailSplitSize;
            reduceSize = tiling.tailReduceSize;
            reduceParam.originalSrcM = tiling.tailM;
            reduceParam.srcM = tiling.tailM;
            reduceParam.dstM = tiling.tailM;
            PipeBarrier<PIPE_V>();
        }
    }
}

__aicore__ inline void SoftmaxFlashV2NoUpdateImpl(const LocalTensor<half>& dst, const LocalTensor<float>& expSumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const ReduceLastND& reduceParam, const SoftMaxTiling& tiling, const uint32_t& offset1, const uint32_t& offset2,
    const uint32_t& splitSize, const uint32_t& reduceSize)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize];

    Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
    PipeBarrier<PIPE_V>();
    NewReduceMaxLastNDImpl(maxTensor[offset2], tmpBuffer0, tmpBuffer1, reduceParam);
    PipeBarrier<PIPE_V>();
    GenericSubNDImpl(tmpBuffer0, tmpBuffer0, maxTensor[offset2], reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);
    PipeBarrier<PIPE_V>();
    Exp(tmpBuffer0, tmpBuffer0, splitSize);
    PipeBarrier<PIPE_V>();
    Cast(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, splitSize);
    PipeBarrier<PIPE_V>();
    NewReduceSumLastNDImpl(expSumTensor[offset2], tmpBuffer0, tmpBuffer1, reduceParam);
}

__aicore__ inline void SoftmaxFlashV2NoUpdateExtImpl(const LocalTensor<half>& dst,
    const LocalTensor<float>& expSumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& src,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling,
    ReduceLastND& reduceParam)
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitSize = tiling.splitSize;
    uint32_t reduceSize = tiling.reduceSize;
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i <= tiling.rangeM; i++) {
        SoftmaxFlashV2NoUpdateImpl(dst, expSumTensor, maxTensor, src, workLocal, reduceParam, tiling, offset1,
            offset2, splitSize, reduceSize);
        offset1 += tiling.splitSize;
        offset2 += tiling.reduceSize;
        if (i == (tiling.rangeM - 1)) {
            if (tiling.tailM == 0) {
                break;
            }
            offset2 = tiling.rangeM * tiling.reduceSize;
            offset1 = tiling.rangeM * tiling.splitSize;
            splitSize = tiling.tailSplitSize;
            reduceSize = tiling.tailReduceSize;
            reduceParam.originalSrcM = tiling.tailM;
            reduceParam.srcM = tiling.tailM;
            reduceParam.dstM = tiling.tailM;
            PipeBarrier<PIPE_V>();
        }
    }
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_NO_UPDATE_IMPL_H