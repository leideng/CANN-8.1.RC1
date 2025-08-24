/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_generic_nz_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_GENERIC_NZ_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_GENERIC_NZ_IMPL_H

namespace AscendC {

template <bool isFlashV2 = false>
__aicore__ inline void SoftMaxGenericNZImpl(const LocalTensor<half>& dst, const LocalTensor<half>& sumTensor,
    const LocalTensor<half>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, uint64_t mask[2], const uint32_t& offset1, const uint32_t& offset2,
    const uint32_t& splitCount, const ReduceLastND& reduceParam)
{
    LocalTensor<float> tmpBuffer0 = workLocal;
    LocalTensor<float> tmpBuffer1 = workLocal[tiling.splitSize];
    const uint32_t splitOffset = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitNZBlockCount = tiling.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t lastSplitNZBlockOffset = splitOffset * (splitNZBlockCount - 1);
    const uint32_t lastBlockMaskLen = reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT : SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<float, half, false>(tmpBuffer0[splitOffset * j],
            src[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();

    PipeBarrier<PIPE_V>();
    ReduceMaxLastNZImpl(tmpBuffer1, tmpBuffer0, mask, reduceParam);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    Cast<half, float, false>(maxTensor[offset2], tmpBuffer1, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });

    for (uint32_t j = 0; j < splitNZBlockCount - 1; j++) {
        Sub<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], tmpBuffer1, MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
    BinaryComputeWithSpecialMask(tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer1,
        mask, lastBlockMaskLen, splitCount, Sub<float>);

    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);

    for (uint32_t j = 0; j < splitNZBlockCount - 1; j++) {
        Exp<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
    UnaryComputeWithSpecialMask(tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer0[lastSplitNZBlockOffset], mask,
        lastBlockMaskLen, splitCount, Exp<float>);

    PipeBarrier<PIPE_V>();
    ReduceSumLastNZImpl(tmpBuffer1, tmpBuffer0, mask, reduceParam);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    Cast<half, float, false>(sumTensor[offset2], tmpBuffer1, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    if constexpr (!isFlashV2) {
        for (uint32_t j = 0; j < splitNZBlockCount - 1; j++) {
            Div<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], tmpBuffer1, MASK_PLACEHOLDER, 1,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        }
        SetMaskNorm();
        ResetMask();
        BinaryComputeWithSpecialMask(tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer1,
            mask, lastBlockMaskLen, splitCount, Div<float>);

        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    }
    PipeBarrier<PIPE_V>();

    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<half, float, false>(dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            tmpBuffer0[splitOffset * j], FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
}

template <bool isFlashV2 = false>
__aicore__ inline void SoftMaxGenericNZImpl(const LocalTensor<float>& dst, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<float>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, uint64_t mask[2], const uint32_t& offset1, const uint32_t& offset2,
    const uint32_t& splitCount, const ReduceLastND& reduceParam)
{
    const uint32_t splitOffset = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitNZBlockCount = tiling.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t lastSplitNZBlockOffset = splitOffset * (splitNZBlockCount - 1);
    const uint32_t lastBlockMaskLen = reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT : SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint16_t copyBlockCount = splitCount / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    LocalTensor<float> tmpBuffer0 = workLocal;
    LocalTensor<float> tmpBuffer1 = workLocal[tiling.splitSize];

    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        DataCopy(tmpBuffer0[splitOffset * j], src[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            splitCount);
    }
    PipeBarrier<PIPE_V>();

    ReduceMaxLastNZImpl(tmpBuffer1, tmpBuffer0, mask, reduceParam);
    PipeBarrier<PIPE_V>();

    DataCopy(maxTensor[offset2], tmpBuffer1, { copyBlockCount, 1, 1, 0 });
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    for (uint32_t j = 0; j < splitNZBlockCount - 1; j++) {
        Sub<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], tmpBuffer1, MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
    BinaryComputeWithSpecialMask(tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer1,
        mask, lastBlockMaskLen, splitCount, Sub<float>);

    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);

    for (uint32_t j = 0; j < splitNZBlockCount - 1; j++) {
        Exp<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
    UnaryComputeWithSpecialMask(tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer0[lastSplitNZBlockOffset], mask,
        lastBlockMaskLen, splitCount, Exp<float>);

    PipeBarrier<PIPE_V>();
    ReduceSumLastNZImpl(tmpBuffer1, tmpBuffer0, mask, reduceParam);
    PipeBarrier<PIPE_V>();

    DataCopy(sumTensor[offset2], tmpBuffer1, { copyBlockCount, 1, 1, 0 });
    PipeBarrier<PIPE_V>();

    if constexpr (isFlashV2) {
        for (uint32_t j = 0; j < splitNZBlockCount; j++) {
            DataCopy(dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], tmpBuffer0[splitOffset * j],
                splitCount);
        }
    } else {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
        for (uint32_t j = 0; j < splitNZBlockCount - 1; j++) {
            Div<float, false>(dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
                tmpBuffer0[splitOffset * j], tmpBuffer1, MASK_PLACEHOLDER, 1,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        }
        SetMaskNorm();
        ResetMask();
        BinaryComputeWithSpecialMask(
            dst[offset1 + (splitNZBlockCount - 1) * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer1, mask, lastBlockMaskLen, splitCount, Div<float>);
    }
}

template <bool isFlashV2 = false>
__aicore__ inline void SoftMaxGenericNZImpl(const LocalTensor<half>& dst, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, uint64_t mask[2], const uint32_t& offset1, const uint32_t& offset2,
    const uint32_t& splitCount, const ReduceLastND& reduceParam)
{
    LocalTensor<float> tmpBuffer0 = workLocal;
    LocalTensor<float> tmpBuffer1 = workLocal[tiling.splitSize];
    const uint32_t lastBlockMaskLen = reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT : SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitOffset = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitNZBlockCount = tiling.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t lastSplitNZBlockOffset = splitOffset * (splitNZBlockCount - 1);
    const uint16_t copyBlockCount = splitCount / SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<float, half, false>(tmpBuffer0[splitOffset * j],
            src[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();

    ReduceMaxLastNZImpl(tmpBuffer1, tmpBuffer0, mask, reduceParam);
    PipeBarrier<PIPE_V>();
    DataCopy(maxTensor[offset2], tmpBuffer1, { copyBlockCount, 1, 1, 0 });
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    for (uint32_t j = 0; j < splitNZBlockCount - 1; j++) {
        Sub<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], tmpBuffer1, MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
    BinaryComputeWithSpecialMask(tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer1,
        mask, lastBlockMaskLen, splitCount, Sub<float>);

    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);

    for (uint32_t j = 0; j < splitNZBlockCount - 1; j++) {
        Exp<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
    UnaryComputeWithSpecialMask(tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer0[lastSplitNZBlockOffset], mask,
        lastBlockMaskLen, splitCount, Exp<float>);

    PipeBarrier<PIPE_V>();
    ReduceSumLastNZImpl(tmpBuffer1, tmpBuffer0, mask, reduceParam);
    PipeBarrier<PIPE_V>();
    DataCopy(sumTensor[offset2], tmpBuffer1, { copyBlockCount, 1, 1, 0 });
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);

    if constexpr (!isFlashV2) {
        for (uint32_t j = 0; j < splitNZBlockCount - 1; j++) {
            Div<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], tmpBuffer1, MASK_PLACEHOLDER, 1,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        }
        SetMaskNorm();
        ResetMask();
        BinaryComputeWithSpecialMask(tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer0[lastSplitNZBlockOffset], tmpBuffer1,
            mask, lastBlockMaskLen, splitCount, Div<float>);

        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    }
    PipeBarrier<PIPE_V>();

    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<half, float, false>(dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            tmpBuffer0[splitOffset * j], FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
}

template <typename T1, typename T2, bool isBasicBlock = false>
__aicore__ inline void SoftMaxNZImpl(const LocalTensor<T1>& dst, const LocalTensor<T1>& sumTensor,
    const LocalTensor<T1>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    const ReduceLastND& mainReduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
        tiling.splitK, tiling.splitM,      SOFTMAX_SHAPE_NZ_BASIC_COUNT };
    const ReduceLastND& tailReduceParam = { tiling.tailM,  originalSrcShape.k, tiling.splitM,
        tiling.splitK, tiling.splitM,      SOFTMAX_SHAPE_NZ_BASIC_COUNT };
    const uint32_t lastBlockMaskLen = originalSrcShape.k % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        originalSrcShape.k % SOFTMAX_SHAPE_NZ_BASIC_COUNT : SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint64_t mask[2] = { 0, 0 };
    CreateSpecialFormatMask(mask[0], lastBlockMaskLen, FLOAT_REPEAT_SIZE / SOFTMAX_SHAPE_NZ_BASIC_COUNT);

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitCount = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset1 = i * splitCount;
        offset2 = i * tiling.reduceSize;
        SoftMaxGenericNZImpl(dst, sumTensor, maxTensor, src, workLocal, tiling, mask, offset1, offset2, splitCount,
            mainReduceParam);
    }
    PipeBarrier<PIPE_V>();
    if (tiling.tailM != 0) {
        offset1 = tiling.rangeM * splitCount;
        offset2 = tiling.rangeM * tiling.reduceSize;
        splitCount = tiling.tailM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        SoftMaxGenericNZImpl(dst, sumTensor, maxTensor, src, workLocal, tiling, mask, offset1, offset2, splitCount,
            tailReduceParam);
    }
}

template <typename T1, typename T2, bool isBasicBlock = false>
__aicore__ inline void SoftMaxNZImpl(const LocalTensor<half>& dst, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    const ReduceLastND& mainReduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
        tiling.splitK, tiling.splitM,      SOFTMAX_SHAPE_NZ_BASIC_COUNT };
    const ReduceLastND& tailReduceParam = { tiling.tailM,  originalSrcShape.k, tiling.splitM,
        tiling.splitK, tiling.splitM,      SOFTMAX_SHAPE_NZ_BASIC_COUNT };
    uint32_t lastBlockMaskLen = originalSrcShape.k % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        originalSrcShape.k % SOFTMAX_SHAPE_NZ_BASIC_COUNT :
        SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint64_t mask[2] = { 0, 0 };
    CreateSpecialFormatMask(mask[0], lastBlockMaskLen, FLOAT_REPEAT_SIZE / SOFTMAX_SHAPE_NZ_BASIC_COUNT);

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitCount = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset1 = i * splitCount;
        offset2 = i * tiling.reduceSize;
        SoftMaxGenericNZImpl(dst, sumTensor, maxTensor, src, workLocal, tiling, mask, offset1, offset2, splitCount,
            mainReduceParam);
    }
    PipeBarrier<PIPE_V>();
    if (tiling.tailM != 0) {
        offset1 = tiling.rangeM * splitCount;
        offset2 = tiling.rangeM * tiling.reduceSize;
        splitCount = tiling.tailM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        SoftMaxGenericNZImpl(dst, sumTensor, maxTensor, src, workLocal, tiling, mask, offset1, offset2, splitCount,
            tailReduceParam);
    }
}

}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_GENERIC_NZ_IMPL_H