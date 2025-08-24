/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_grad_nz_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_COMMON_SOFTMAX_GRAD_NZ_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_COMMON_SOFTMAX_GRAD_NZ_IMPL_H


namespace AscendC {

__aicore__ inline void SoftMaxGradFrontGenericNZImpl(const LocalTensor<half>& dst, const LocalTensor<half>& gradTensor,
    const LocalTensor<half>& src, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling, uint64_t mask[2],
    const uint32_t& offset1, const uint32_t& offset2, const uint32_t& splitCount, const ReduceLastND& reduceParam)
{
    LocalTensor<float> tmpBuffer0 = workLocal;
    LocalTensor<float> tmpBuffer1 = workLocal[tiling.splitSize];
    LocalTensor<float> tmpBuffer2 = workLocal[tiling.splitSize + tiling.splitSize];
    const uint32_t splitNZBlockCount = tiling.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitOffset = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<float, half, false>(tmpBuffer0[splitOffset * j],
            src[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
        Cast<float, half, false>(tmpBuffer1[splitOffset * j],
            gradTensor[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], RoundMode::CAST_NONE,
            MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
    }

    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Mul<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], tmpBuffer1[splitOffset * j],
            MASK_PLACEHOLDER, 1, { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();
    ReduceSumLastNZImpl(tmpBuffer2, tmpBuffer0, mask, reduceParam);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    Cast<half, float, false>(dst[offset2], tmpBuffer2, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void SoftMaxGradFrontGenericNZImpl(const LocalTensor<float>& dst,
    const LocalTensor<float>& gradTensor, const LocalTensor<float>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, uint64_t mask[2], const uint32_t& offset1, const uint32_t& offset2,
    const uint32_t& splitCount, const ReduceLastND& reduceParam)
{
    LocalTensor<float> tmpBuffer0 = workLocal;
    LocalTensor<float> tmpBuffer1 = workLocal[tiling.splitSize];
    const uint32_t splitOffset = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitNZBlockCount = tiling.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);

    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Mul<float, false>(tmpBuffer0[splitOffset * j], src[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            gradTensor[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();

    PipeBarrier<PIPE_V>();
    ReduceSumLastNZImpl(tmpBuffer1, tmpBuffer0, mask, reduceParam);
    PipeBarrier<PIPE_V>();
    // out datacopy 64B->32B
    DataCopy(dst[offset2], tmpBuffer1, { (uint16_t)reduceParam.originalSrcM, 1, 1, 0 });
}

template <typename T>
__aicore__ inline void SoftmaxGradFrontNZImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape,
    const SoftMaxTiling& tiling)
{
    const ReduceLastND& mainReduceParam = { tiling.splitM, tiling.splitK, tiling.splitM,
        tiling.splitK, tiling.splitM, SOFTMAX_SHAPE_NZ_BASIC_COUNT };
    const ReduceLastND& tailReduceParam = { tiling.tailM,  tiling.splitK, tiling.splitM,
        tiling.splitK, tiling.splitM, SOFTMAX_SHAPE_NZ_BASIC_COUNT };

    const uint32_t lastBlockMaskLen = tiling.splitK % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        tiling.splitK % SOFTMAX_SHAPE_NZ_BASIC_COUNT :
        SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint64_t mask[2] = { 0, 0 };
    CreateSpecialFormatMask(mask[0], lastBlockMaskLen, FLOAT_REPEAT_SIZE / SOFTMAX_SHAPE_NZ_BASIC_COUNT);

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitCount = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset1 = i * splitCount;
        offset2 = i * tiling.reduceSize;
        SoftMaxGradFrontGenericNZImpl(dstTensor, gradTensor, srcTensor, workLocal, tiling, mask, offset1, offset2,
            splitCount, mainReduceParam);
    }
    PipeBarrier<PIPE_V>();
    if (tiling.tailM != 0) {
        offset1 = tiling.rangeM * splitCount;
        offset2 = tiling.rangeM * tiling.reduceSize;
        splitCount = tiling.tailM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        SoftMaxGradFrontGenericNZImpl(dstTensor, gradTensor, srcTensor, workLocal, tiling, mask, offset1, offset2,
            splitCount, tailReduceParam);
    }
}

__aicore__ inline void SoftMaxGradGenericNZImpl(const LocalTensor<half>& dst, const LocalTensor<half>& gradTensor,
    const LocalTensor<half>& src, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling, uint64_t mask[2],
    const uint32_t& offset, const uint32_t& splitCount, const ReduceLastND& reduceParam)
{
    LocalTensor<float> tmpBuffer0 = workLocal;
    LocalTensor<float> tmpBuffer1 = workLocal[tiling.splitSize];
    LocalTensor<float> tmpBuffer2 = workLocal[tiling.splitSize + tiling.splitSize];
    LocalTensor<float> tmpBuffer3 = workLocal[tiling.splitSize + tiling.splitSize + tiling.splitSize];
    const uint32_t splitNZBlockCount = tiling.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitOffset = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);

    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<float, half, false>(tmpBuffer0[splitOffset * j],
            src[offset + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
        Cast<float, half, false>(tmpBuffer1[splitOffset * j],
            gradTensor[offset + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], RoundMode::CAST_NONE, MASK_PLACEHOLDER,
            1, { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
    }

    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Mul<float, false>(tmpBuffer2[splitOffset * j], tmpBuffer0[splitOffset * j], tmpBuffer1[splitOffset * j],
            MASK_PLACEHOLDER, 1, { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();

    PipeBarrier<PIPE_V>();
    ReduceSumLastNZImpl(tmpBuffer3, tmpBuffer2, mask, reduceParam);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);

    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Sub<float, false>(tmpBuffer1[splitOffset * j], tmpBuffer1[splitOffset * j], tmpBuffer3, MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }

    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Mul<float, false>(tmpBuffer2[splitOffset * j], tmpBuffer1[splitOffset * j], tmpBuffer0[splitOffset * j],
            MASK_PLACEHOLDER, 1, { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }

    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<half, float, false>(dst[offset + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            tmpBuffer2[splitOffset * j], FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }

    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void SoftMaxGradGenericNZImpl(const LocalTensor<float>& dst, const LocalTensor<float>& gradTensor,
    const LocalTensor<float>& src, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling, uint64_t mask[2],
    const uint32_t& offset, const uint32_t& splitCount, const ReduceLastND& reduceParam)
{
    LocalTensor<float> tmpBuffer0 = workLocal;
    LocalTensor<float> tmpBuffer1 = workLocal[tiling.splitSize];
    const uint32_t splitNZBlockCount = tiling.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitOffset = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Mul<float, false>(tmpBuffer0[splitOffset * j], src[offset + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            gradTensor[offset + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();

    PipeBarrier<PIPE_V>();
    ReduceSumLastNZImpl(tmpBuffer1, tmpBuffer0, mask, reduceParam);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Sub<float, false>(tmpBuffer0[splitOffset * j],
            gradTensor[offset + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], tmpBuffer1, MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Mul<float, false>(dst[offset + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], tmpBuffer0[splitOffset * j],
            src[offset + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void SoftmaxGradNZImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape,
    const SoftMaxTiling& tiling, bool isFront = false)
{
    if (isFront) {
        SoftmaxGradFrontNZImpl(dstTensor, gradTensor, srcTensor, workLocal, originalSrcShape, tiling);
    } else {
        const ReduceLastND& mainReduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
            tiling.splitK, tiling.splitM,      SOFTMAX_SHAPE_NZ_BASIC_COUNT };
        const ReduceLastND& tailReduceParam = { tiling.tailM,  originalSrcShape.k, tiling.splitM,
            tiling.splitK, tiling.splitM,      SOFTMAX_SHAPE_NZ_BASIC_COUNT };
        uint32_t lastBlockMaskLen = tiling.splitK % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
            tiling.splitK % SOFTMAX_SHAPE_NZ_BASIC_COUNT :
            SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        uint64_t mask[2] = { 0, 0 };
        CreateSpecialFormatMask(mask[0], lastBlockMaskLen, FLOAT_REPEAT_SIZE / SOFTMAX_SHAPE_NZ_BASIC_COUNT);
        uint32_t offset = 0;
        uint32_t splitCount = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;

        for (uint32_t i = 0; i < tiling.rangeM; i++) {
            offset = i * splitCount;
            SoftMaxGradGenericNZImpl(dstTensor, gradTensor, srcTensor, workLocal, tiling, mask, offset, splitCount,
                mainReduceParam);
        }
        PipeBarrier<PIPE_V>();
        if (tiling.tailM != 0) {
            offset = tiling.rangeM * splitCount;
            splitCount = tiling.tailM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
            SoftMaxGradGenericNZImpl(dstTensor, gradTensor, srcTensor, workLocal, tiling, mask, offset, splitCount,
                tailReduceParam);
        }
    }
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_COMMON_SOFTMAX_GRAD_NZ_IMPL_H