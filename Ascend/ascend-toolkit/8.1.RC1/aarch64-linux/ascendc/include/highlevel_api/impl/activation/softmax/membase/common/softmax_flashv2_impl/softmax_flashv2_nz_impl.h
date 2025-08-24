/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_flashv2_nz_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_NZ_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_NZ_IMPL_H

namespace AscendC {

__aicore__ inline void FlashV2NZUpdateGenericImpl(const LocalTensor<float>& dst, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<float>& src, const LocalTensor<float>& expMaxTensor,
    const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, uint64_t mask[2], const uint32_t& offset1, const uint32_t& offset2,
    const uint32_t& splitCount, const ReduceLastND& reduceParam)
{
    LocalTensor<float> tmpBuffer0 = workLocal;
    LocalTensor<float> tmpBuffer1 = workLocal[tiling.splitSize]; // default len splitM * 16
    LocalTensor<float> inMaxTmp = workLocal[tiling.splitSize + tiling.reduceSize + tiling.reduceSize];
    LocalTensor<float> inSumTmp =
        workLocal[tiling.splitSize + tiling.reduceSize + tiling.reduceSize + tiling.reduceSize];
    const uint32_t splitOffset = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint16_t copyBlockCount = splitCount / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitNZBlockCount = tiling.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t lastSplitNZBlockOffset = splitOffset * (splitNZBlockCount - 1);
    const uint32_t lastBlockMaskLen = reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT :
        SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    DataCopy(inSumTmp, inSumTensor[offset2], { 1, copyBlockCount, 0, 0 });
    DataCopy(inMaxTmp, inMaxTensor[offset2], { 1, copyBlockCount, 0, 0 });

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
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount / B16_BYTE_SIZE);
    Max<float, false>(maxTensor[offset2], inMaxTmp, maxTensor[offset2], MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Sub<float, false>(inMaxTmp, inMaxTmp, maxTensor[offset2], MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Exp<float, false>(expMaxTensor[offset2], inMaxTmp, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();

    DataCopy(tmpBuffer1, maxTensor[offset2], { copyBlockCount, 1, 0, 1 });
    DataCopy(tmpBuffer1[FLOAT_NUM_PER_BLK], maxTensor[offset2], { copyBlockCount, 1, 0, 1 });

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
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        DataCopy(dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], tmpBuffer0[splitOffset * j],
            splitCount);
    }

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount / B16_BYTE_SIZE);
    // update sum = expmax * insum + sum
    Mul<float, false>(inMaxTmp, expMaxTensor[offset2], inSumTmp, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Add<float, false>(sumTensor[offset2], inMaxTmp, sumTensor[offset2], MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void FlashV2NZUpdateGenericImpl(const LocalTensor<half>& dst, const LocalTensor<half>& sumTensor,
    const LocalTensor<half>& maxTensor, const LocalTensor<half>& src, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<half>& inSumTensor, const LocalTensor<half>& inMaxTensor, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, uint64_t mask[2], const uint32_t& offset1, const uint32_t& offset2,
    const uint32_t& splitCount, const ReduceLastND& reduceParam)
{
    LocalTensor<float> tmpBuffer0 = workLocal;
    LocalTensor<float> tmpBuffer1 = workLocal[tiling.splitSize];
    LocalTensor<float> inMaxTmp = workLocal[tiling.splitSize + tiling.reduceSize];
    LocalTensor<float> inSumTmp = workLocal[tiling.splitSize + tiling.reduceSize + tiling.reduceSize];
    const uint32_t splitOffset = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitNZBlockCount = tiling.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t lastSplitNZBlockOffset = splitOffset * (splitNZBlockCount - 1);
    const uint32_t lastBlockMaskLen = reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT :
        SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<float, half, false>(tmpBuffer0[splitOffset * j],
            src[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
    }
    Cast<float, half, false>(inMaxTmp, inMaxTensor[offset2], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
    Cast<float, half, false>(inSumTmp, inSumTensor[offset2], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
    SetMaskNorm();
    ResetMask();

    ReduceMaxLastNZImpl(tmpBuffer1, tmpBuffer0, mask, reduceParam);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);

    Max<float, false>(tmpBuffer1, inMaxTmp, tmpBuffer1, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Sub<float, false>(inMaxTmp, inMaxTmp, tmpBuffer1, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Exp<float, false>(inMaxTmp, inMaxTmp, MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Cast<half, float, false>(expMaxTensor[offset2], inMaxTmp, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
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

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);

    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<half, float, false>(dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            tmpBuffer0[splitOffset * j], FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    Mul<float, false>(inMaxTmp, inMaxTmp, inSumTmp, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Add<float, false>(inSumTmp, inMaxTmp, tmpBuffer1, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Cast<half, float, false>(sumTensor[offset2], inSumTmp, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void FlashV2NZUpdateGenericImpl(const LocalTensor<half>& dst, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, uint64_t mask[2], const uint32_t& offset1, const uint32_t& offset2,
    const uint32_t& splitCount, const ReduceLastND& reduceParam)
{
    LocalTensor<float> tmpBuffer0 = workLocal;
    LocalTensor<float> tmpBuffer1 = workLocal[tiling.splitSize]; // default len splitM * 16
    LocalTensor<float> inMaxTmp = workLocal[tiling.splitSize + tiling.reduceSize + tiling.reduceSize];
    LocalTensor<float> inSumTmp =
        workLocal[tiling.splitSize + tiling.reduceSize + tiling.reduceSize + tiling.reduceSize];
    const uint32_t splitOffset = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitNZBlockCount = tiling.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t lastSplitNZBlockOffset = splitOffset * (splitNZBlockCount - 1);
    const uint32_t lastBlockMaskLen = reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT :
        SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint16_t copyBlockCount = splitCount / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    DataCopy(inMaxTmp, inMaxTensor[offset2], { 1, copyBlockCount, 0, 0 });
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
    // 64B to 32B
    DataCopy(maxTensor[offset2], tmpBuffer1, { copyBlockCount, 1, 1, 0 });
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount / B16_BYTE_SIZE);
    Max<float, false>(maxTensor[offset2], inMaxTmp, maxTensor[offset2], MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Sub<float, false>(inMaxTmp, inMaxTmp, maxTensor[offset2], MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Exp<float, false>(inMaxTmp, inMaxTmp, MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();

    DataCopy(tmpBuffer1, inMaxTmp, { copyBlockCount, 1, 0, 1 });
    DataCopy(tmpBuffer1[FLOAT_NUM_PER_BLK], inMaxTmp, { copyBlockCount, 1, 0, 1 });

    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    Cast<half, float, false>(expMaxTensor[offset2 * B16_BYTE_SIZE], tmpBuffer1, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER,
        1, { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    SetMaskNorm();
    ResetMask();

    DataCopy(tmpBuffer1, maxTensor[offset2], { copyBlockCount, 1, 0, 1 });
    DataCopy(tmpBuffer1[FLOAT_NUM_PER_BLK], maxTensor[offset2], { copyBlockCount, 1, 0, 1 });

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
    DataCopy(inSumTmp, inSumTensor[offset2], { 1, copyBlockCount, 0, 0 });

    ReduceSumLastNZImpl(tmpBuffer1, tmpBuffer0, mask, reduceParam);
    PipeBarrier<PIPE_V>();

    DataCopy(sumTensor[offset2], tmpBuffer1, { copyBlockCount, 1, 1, 0 });
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<half, float, false>(dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            tmpBuffer0[splitOffset * j], FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }

    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount / B16_BYTE_SIZE);
    // update sum = expmax * insum + sum
    Mul<float, false>(inMaxTmp, inMaxTmp, inSumTmp, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Add<float, false>(sumTensor[offset2], inMaxTmp, sumTensor[offset2], MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    SetMaskNorm();
    ResetMask();
}

template <typename T1, typename T2, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashV2NZUpdateImpl(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
    const LocalTensor<T2>& inSumTensor, const LocalTensor<T2>& inMaxTensor, const LocalTensor<float>& workLocal,
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
    uint32_t paddingTailCount = (tiling.srcM - originalSrcShape.m) * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset1 = i * splitCount;
        offset2 = i * tiling.reduceSize;
        if (tiling.tailM == 0 && i == tiling.rangeM - 1 && splitCount > paddingTailCount) {
            splitCount -= paddingTailCount;
        }
        FlashV2NZUpdateGenericImpl(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor,
            workLocal, tiling, mask, offset1, offset2, splitCount, mainReduceParam);
    }
    PipeBarrier<PIPE_V>();
    if (tiling.tailM != 0) {
        offset1 = tiling.rangeM * splitCount;
        offset2 = tiling.rangeM * tiling.reduceSize;
        splitCount = tiling.tailM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        if (splitCount > paddingTailCount) {
            splitCount -= paddingTailCount;
        }
        FlashV2NZUpdateGenericImpl(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor,
            workLocal, tiling, mask, offset1, offset2, splitCount, tailReduceParam);
    }
}

template <typename T1, typename T2, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashV2NZUpdateImpl(const LocalTensor<half>& dstTensor,
    const LocalTensor<float>& sumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor,
    const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
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
    uint32_t paddingTailCount = (tiling.srcM - originalSrcShape.m) * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset1 = i * splitCount;
        offset2 = i * tiling.reduceSize;
        if (tiling.tailM == 0 && i == tiling.rangeM - 1 && splitCount > paddingTailCount) {
            splitCount -= paddingTailCount;
        }
        FlashV2NZUpdateGenericImpl(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor,
            workLocal, tiling, mask, offset1, offset2, splitCount, mainReduceParam);
    }
    PipeBarrier<PIPE_V>();
    if (tiling.tailM != 0) {
        offset1 = tiling.rangeM * splitCount;
        offset2 = tiling.rangeM * tiling.reduceSize;
        splitCount = tiling.tailM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        if (splitCount > paddingTailCount) {
            splitCount -= paddingTailCount;
        }
        FlashV2NZUpdateGenericImpl(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor,
            workLocal, tiling, mask, offset1, offset2, splitCount, tailReduceParam);
    }
}

template <typename T1, typename T2, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashV2NZNoUpdateImpl(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    const ReduceLastND& mainReduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
        tiling.splitK, tiling.splitM,      SOFTMAX_SHAPE_NZ_BASIC_COUNT };
    const ReduceLastND& tailReduceParam = { tiling.tailM,  originalSrcShape.k, tiling.splitM,
        tiling.splitK, tiling.splitM,      SOFTMAX_SHAPE_NZ_BASIC_COUNT };
    const uint32_t lastBlockMaskLen = originalSrcShape.k % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        originalSrcShape.k % SOFTMAX_SHAPE_NZ_BASIC_COUNT :
        SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint64_t mask[2] = { 0, 0 };
    CreateSpecialFormatMask(mask[0], lastBlockMaskLen, FLOAT_REPEAT_SIZE / SOFTMAX_SHAPE_NZ_BASIC_COUNT);

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitCount = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint32_t paddingTailCount = (tiling.srcM - originalSrcShape.m) * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset1 = i * splitCount;
        offset2 = i * tiling.reduceSize;
        if (tiling.tailM == 0 && i == tiling.rangeM - 1 && splitCount > paddingTailCount) {
            splitCount -= paddingTailCount;
        }
        SoftMaxGenericNZImpl<true>(dstTensor, sumTensor, maxTensor, srcTensor, workLocal, tiling, mask, offset1,
            offset2, splitCount, mainReduceParam);
    }
    PipeBarrier<PIPE_V>();
    if (tiling.tailM != 0) {
        offset1 = tiling.rangeM * splitCount;
        offset2 = tiling.rangeM * tiling.reduceSize;
        splitCount = tiling.tailM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        if (splitCount > paddingTailCount) {
            splitCount -= paddingTailCount;
        }
        SoftMaxGenericNZImpl<true>(dstTensor, sumTensor, maxTensor, srcTensor, workLocal, tiling, mask, offset1,
            offset2, splitCount, tailReduceParam);
    }
}

template <typename T1, typename T2, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashV2NZNoUpdateImpl(const LocalTensor<half>& dstTensor,
    const LocalTensor<float>& sumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
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
    uint32_t paddingTailCount = (tiling.srcM - originalSrcShape.m) * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset1 = i * splitCount;
        offset2 = i * tiling.reduceSize;
        if (tiling.tailM == 0 && i == tiling.rangeM - 1 && splitCount > paddingTailCount) {
            splitCount -= paddingTailCount;
        }
        SoftMaxGenericNZImpl<true>(dstTensor, sumTensor, maxTensor, srcTensor, workLocal, tiling, mask, offset1,
            offset2, splitCount, mainReduceParam);
    }
    PipeBarrier<PIPE_V>();
    if (tiling.tailM != 0) {
        offset1 = tiling.rangeM * splitCount;
        offset2 = tiling.rangeM * tiling.reduceSize;
        splitCount = tiling.tailM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        if (splitCount > paddingTailCount) {
            splitCount -= paddingTailCount;
        }
        SoftMaxGenericNZImpl<true>(dstTensor, sumTensor, maxTensor, srcTensor, workLocal, tiling, mask, offset1,
            offset2, splitCount, tailReduceParam);
    }
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_NZ_IMPL_H