/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_generic_nd_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_GENERIC_ND_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_GENERIC_ND_IMPL_H

namespace AscendC {

__aicore__ inline void SoftMaxGenericNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<float>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const uint32_t& offset1, const uint32_t& offset2, const uint32_t& splitSize,
    const uint32_t& reduceSize, const ReduceLastND& reduceParam)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal; // need splitM * 64

    NewReduceMaxLastNDImpl(maxTensor[offset2], src[offset1], tmpBuffer0, reduceParam);
    PipeBarrier<PIPE_V>();

    GenericSubNDImpl(dst[offset1], src[offset1], maxTensor[offset2], reduceParam.originalSrcM, tiling.srcK,
        tiling.reduceK);

    PipeBarrier<PIPE_V>();
    Exp(dst[offset1], dst[offset1], splitSize);
    PipeBarrier<PIPE_V>();
    NewReduceSumLastNDImpl(sumTensor[offset2], dst[offset1], tmpBuffer0, reduceParam);
    PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
    TransDivToMulImpl(dst[offset1], sumTensor[offset2], tmpBuffer0, reduceParam.originalSrcM, tiling.srcK,
        tiling.reduceK);
#elif __CCE_AICORE__ == 220
    GenericDivNDImpl(dst[offset1], dst[offset1], sumTensor[offset2], reduceParam.originalSrcM, tiling.srcK,
        tiling.reduceK);
#endif
}

__aicore__ inline void SoftMaxGenericNDImpl(const LocalTensor<half>& dst, const LocalTensor<half>& sumTensor,
    const LocalTensor<half>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const uint32_t& offset1, const uint32_t& offset2, const uint32_t& splitSize,
    const uint32_t& reduceSize, const ReduceLastND& reduceParam)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitSize];
    const LocalTensor<float>& tmpBuffer3 = workLocal[tiling.splitSize + tiling.reduceSize]; // need splitM * 64
    Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
    PipeBarrier<PIPE_V>();
    NewReduceMaxLastNDImpl(tmpBuffer2, tmpBuffer0, tmpBuffer3, reduceParam);

    PipeBarrier<PIPE_V>();
    Cast(maxTensor[offset2], tmpBuffer2, FLOAT2HALF_ROUND_MODE, reduceSize);
    PipeBarrier<PIPE_V>();

    GenericSubNDImpl(tmpBuffer0, tmpBuffer0, tmpBuffer2, reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);

    PipeBarrier<PIPE_V>();
    Exp(tmpBuffer0, tmpBuffer0, splitSize);
    PipeBarrier<PIPE_V>();

    NewReduceSumLastNDImpl(tmpBuffer2, tmpBuffer0, tmpBuffer3, reduceParam);
    PipeBarrier<PIPE_V>();

    Cast(sumTensor[offset2], tmpBuffer2, FLOAT2HALF_ROUND_MODE, reduceSize);
    PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
    TransDivToMulImpl(tmpBuffer0, tmpBuffer2, tmpBuffer3, reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);
#elif __CCE_AICORE__ == 220
    GenericDivNDImpl(tmpBuffer0, tmpBuffer0, tmpBuffer2, reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);
#endif    
    PipeBarrier<PIPE_V>();
    Cast(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, splitSize);
}

__aicore__ inline void SoftMaxGenericNDImpl(const LocalTensor<half>& dst, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const uint32_t& offset1, const uint32_t& offset2, const uint32_t& splitSize,
    const ReduceLastND& reduceParam)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize]; // need splitM * 64

    Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
    PipeBarrier<PIPE_V>();
    NewReduceMaxLastNDImpl(maxTensor[offset2], tmpBuffer0, tmpBuffer1, reduceParam);
    PipeBarrier<PIPE_V>();

    GenericSubNDImpl(tmpBuffer0, tmpBuffer0, maxTensor[offset2], reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);

    PipeBarrier<PIPE_V>();
    Exp(tmpBuffer0, tmpBuffer0, splitSize);

    PipeBarrier<PIPE_V>();
    NewReduceSumLastNDImpl(sumTensor[offset2], tmpBuffer0, tmpBuffer1, reduceParam);
    PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
    TransDivToMulImpl(tmpBuffer0, sumTensor[offset2], tmpBuffer1, reduceParam.originalSrcM, tiling.srcK,
        tiling.reduceK);
#elif __CCE_AICORE__ == 220
    GenericDivNDImpl(tmpBuffer0, tmpBuffer0, sumTensor[offset2], reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);
#endif
    PipeBarrier<PIPE_V>();
    Cast(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, splitSize);
}

template <typename T>
__aicore__ inline void SoftMaxNDExtImpl(const LocalTensor<T>& dst, const LocalTensor<T>& sumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling, ReduceLastND& reduceParam)
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitSize = tiling.splitSize;
    uint32_t reduceSize = tiling.reduceSize;
    for (uint32_t i = 0; i <= tiling.rangeM; i++) {
        SoftMaxGenericNDImpl(dst, sumTensor, maxTensor, src, workLocal, tiling, offset1, offset2, splitSize,
            reduceSize, reduceParam);
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

__aicore__ inline void SoftMaxNDExtImpl(const LocalTensor<half>& dst, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling, ReduceLastND& reduceParam)
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitSize = tiling.splitSize;
    uint32_t reduceSize = tiling.reduceSize;
    for (uint32_t i = 0; i <= tiling.rangeM; i++) {
        SoftMaxGenericNDImpl(dst, sumTensor, maxTensor, src, workLocal, tiling, offset1, offset2, splitSize,
            reduceParam);
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
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_GENERIC_ND_IMPL_H