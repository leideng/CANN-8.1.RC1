/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_common_nd_reduce.h 
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_COMMON_ND_REDUCE_H
#define IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_COMMON_ND_REDUCE_H

namespace AscendC {

__aicore__ inline void FirstBlockCopyImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t srcM, const uint32_t srcK, const uint16_t dstRepeatStride, const uint16_t srcRepeatStride)
{
#if ASCENDC_CPU_DEBUG == 0
    const uint32_t range = srcM / MAX_REPEAT_TIMES;
    const uint32_t tail = srcM % MAX_REPEAT_TIMES;
    for (uint32_t i = 0; i < range; i++) {
        Copy<float, false>(dst[i * SOFTMAX_MAX_REPEAT_CLC_FLOAT_NUM], src[i * MAX_REPEAT_TIMES * srcK],
            MASK_PLACEHOLDER, MAX_REPEAT_TIMES, { 1, 1, dstRepeatStride, srcRepeatStride });
    }
    if (tail != 0) {
        Copy<float, false>(dst[range * SOFTMAX_MAX_REPEAT_CLC_FLOAT_NUM], src[range * MAX_REPEAT_TIMES * srcK],
            MASK_PLACEHOLDER, tail, { 1, 1, dstRepeatStride, srcRepeatStride });
    }
#else
    for (uint32_t i = 0; i < srcM; i++) {
        DataCopy(dst[i * FLOAT_REPEAT_SIZE], src[i * srcK], FLOAT_REPEAT_SIZE);
    }
#endif
}

__aicore__ inline void ReduceMaxLastNDSplitImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const struct ReduceLastND& reduceParam, uint64_t mask, uint32_t splitNum)
{
    uint32_t range = reduceParam.srcM / MAX_REPEAT_TIMES;
    uint32_t tail = reduceParam.srcM % MAX_REPEAT_TIMES;

    for (uint32_t i = 0; i < range; i++) {
        WholeReduceMax(dst[i * MAX_REPEAT_TIMES],
            src[splitNum * FLOAT_REPEAT_SIZE + i * MAX_REPEAT_TIMES * reduceParam.srcK], mask, MAX_REPEAT_TIMES, 1, 1,
            reduceParam.srcK / FLOAT_NUM_PER_BLK, ReduceOrder::ORDER_ONLY_VALUE);
    }
    if (tail != 0) {
        WholeReduceMax(dst[range * MAX_REPEAT_TIMES],
            src[splitNum * FLOAT_REPEAT_SIZE + range * MAX_REPEAT_TIMES * reduceParam.srcK], mask, tail, 1, 1,
            reduceParam.srcK / FLOAT_NUM_PER_BLK, ReduceOrder::ORDER_ONLY_VALUE);
    }
}

__aicore__ inline void AlignedReduceMaxNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpTensor, const struct ReduceLastND& reduceMaxParam, const uint32_t splitCount)
{
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceMaxParam.srcM * FLOAT_REPEAT_SIZE);
    BlockReduceMax<float, false>(tmpTensor, src, 1, MASK_PLACEHOLDER, 1, 1, reduceMaxParam.srcK / FLOAT_NUM_PER_BLK);
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();
    DataCopy(dst, tmpTensor, { 1, (uint16_t)reduceMaxParam.srcM, 0, 0 });
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    for (uint32_t i = 1; i < splitCount; i++) {
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceMaxParam.srcM * FLOAT_REPEAT_SIZE);
        BlockReduceMax<float, false>(tmpTensor, src[i * FLOAT_REPEAT_SIZE], 1, MASK_PLACEHOLDER, 1, 1,
            reduceMaxParam.srcK / FLOAT_NUM_PER_BLK);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceMaxParam.srcM * FLOAT_NUM_PER_BLK);
        Max<float, false>(dst, dst, tmpTensor, MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
    }
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceMaxParam.srcM * FLOAT_NUM_PER_BLK);
    BlockReduceMax<float, false>(dst, dst, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void AlignedReduceSumNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpTensor, const struct ReduceLastND& reduceParam, const uint32_t splitCount)
{
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_REPEAT_SIZE);
    BlockReduceSum<float, false>(tmpTensor, src, 1, MASK_PLACEHOLDER, 1, 1, reduceParam.srcK / FLOAT_NUM_PER_BLK);
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();
    DataCopy(dst, tmpTensor, { 1, (uint16_t)reduceParam.srcM, 0, 0 });
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    for (uint32_t i = 1; i < splitCount; i++) {
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_REPEAT_SIZE);
        BlockReduceSum<float, false>(tmpTensor, src[i * FLOAT_REPEAT_SIZE], 1, MASK_PLACEHOLDER, 1, 1,
            reduceParam.srcK / FLOAT_NUM_PER_BLK);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_NUM_PER_BLK);
        Add<float, false>(dst, dst, tmpTensor, MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
    }
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_NUM_PER_BLK);
    BlockReduceSum<float, false>(dst, dst, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void ReduceMaxLastNDImpl(const LocalTensor<float>& dstMax, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpTensor, const struct ReduceLastND& reduceMaxParam)
{
    const uint32_t splitCount = reduceMaxParam.originalSrcK / FLOAT_REPEAT_SIZE;
    const uint32_t tailSrcK = reduceMaxParam.originalSrcK % FLOAT_REPEAT_SIZE;
    if (splitCount > 0) {
        AlignedReduceMaxNDImpl(tmpTensor, src, dstMax, reduceMaxParam, splitCount);
    }
    if (tailSrcK != 0) {
        ReduceMaxLastNDSplitImpl(dstMax, src, reduceMaxParam, tailSrcK, splitCount);
        PipeBarrier<PIPE_V>();
        if (splitCount == 0) {
            DataCopy(tmpTensor, dstMax, { 1, (uint16_t)reduceMaxParam.srcM, 0, 0 });
        } else {
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(0, reduceMaxParam.srcM * FLOAT_NUM_PER_BLK);
            Max<float, false>(tmpTensor, tmpTensor, dstMax, MASK_PLACEHOLDER, 1,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
            SetMaskNorm();
            ResetMask();
        }
    }

    PipeBarrier<PIPE_V>();
    SingleBlockBroadCastImpl(dstMax, tmpTensor, reduceMaxParam);
}

__aicore__ inline void BasicBlockReduceMaxImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t originalSrcM, const uint32_t reduceK)
{
    if (originalSrcM == 1) {
        WholeReduceMax<float, false>(dst, src, MASK_PLACEHOLDER, DEFAULT_REPEAT_STRIDE, 1, 1, 0,
            ReduceOrder::ORDER_ONLY_VALUE);
        if (reduceK == DEFAULT_REPEAT_STRIDE * HALF_FACTOR) {
            PipeBarrier<PIPE_V>();
            DataCopy(dst[DEFAULT_REPEAT_STRIDE], dst, { 1, 1, 0, 0 });
        }
    } else {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, originalSrcM * FLOAT_REPEAT_SIZE);
        BlockReduceMax<float, false>(src, src, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(0, originalSrcM * FLOAT_NUM_PER_BLK);
        BlockReduceMax<float, false>(dst, src, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        SetMaskNorm();
        ResetMask();
    }
}

template <bool isBroadCast = true>
__aicore__ inline void NewReduceMaxLastNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpTensor, const struct ReduceLastND& reduceParam)
{
    const uint32_t splitCount = reduceParam.originalSrcK / FLOAT_REPEAT_SIZE;
    const uint32_t tailSrcK = reduceParam.originalSrcK % FLOAT_REPEAT_SIZE;
    const uint16_t srcRepeatStride = reduceParam.srcK / FLOAT_NUM_PER_BLK;

    if (reduceParam.originalSrcK < FLOAT_REPEAT_SIZE) { // reduce axis length is (0, 64)
        ReduceMaxLastNDSplitImpl(dst, src, reduceParam, reduceParam.originalSrcK, 0);
        ResetMask();
    } else {
        if (reduceParam.originalSrcK >= SOFTMAX_FLOAT_SPECIAL_BLOCKREDUCE_LEN) { // reduce axis length >= 512
            BigBlockReduceMax(tmpTensor, src, splitCount, reduceParam.originalSrcM, reduceParam.srcK);
        } else if (reduceParam.originalSrcK >= HALF_REPEAT_SIZE) { // reduce axis length is [128, 512)
            Max<float, false>(tmpTensor, src, src[FLOAT_REPEAT_SIZE], MASK_PLACEHOLDER,
                (uint8_t)(reduceParam.originalSrcM),
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, (uint8_t)srcRepeatStride, (uint8_t)srcRepeatStride });
            NextBlockMaxImpl(tmpTensor, src, (uint8_t)(reduceParam.originalSrcM), srcRepeatStride, splitCount,
                reduceParam.srcK);
        } else { // reduce axis length is [64, 128)
            FirstBlockCopyImpl(tmpTensor, src, reduceParam.originalSrcM, reduceParam.srcK, DEFAULT_REPEAT_STRIDE,
                srcRepeatStride);
        }

        if (tailSrcK != 0) {
            PipeBarrier<PIPE_V>();
            TailMaxImpl(tmpTensor, src, reduceParam, tailSrcK, srcRepeatStride, splitCount);
            ResetMask();
        }
        PipeBarrier<PIPE_V>();
        BasicBlockReduceMaxImpl(dst, tmpTensor, reduceParam.originalSrcM, reduceParam.dstK);
    }
    if constexpr (isBroadCast) {
        if (reduceParam.originalSrcM != 1 || reduceParam.originalSrcK <= FLOAT_REPEAT_SIZE) {
            PipeBarrier<PIPE_V>();
            AlignedBroadCastImpl(dst, tmpTensor, reduceParam);
        }
    }
}

__aicore__ inline void ReduceSumLastNDSplitImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const struct ReduceLastND& reduceParam, uint64_t mask, uint32_t dstRepStride, uint32_t splitNum)
{
    uint32_t range = reduceParam.srcM / MAX_REPEAT_TIMES;
    uint32_t tail = reduceParam.srcM % MAX_REPEAT_TIMES;

    for (uint32_t i = 0; i < range; i++) {
        WholeReduceSum(dst[i * MAX_REPEAT_TIMES],
            src[splitNum * FLOAT_REPEAT_SIZE + i * MAX_REPEAT_TIMES * reduceParam.srcK], mask, MAX_REPEAT_TIMES,
            dstRepStride, 1,
            reduceParam.srcK / FLOAT_NUM_PER_BLK);
    }
    if (tail != 0) {
        WholeReduceSum(dst[range * MAX_REPEAT_TIMES],
            src[splitNum * FLOAT_REPEAT_SIZE + range * MAX_REPEAT_TIMES * reduceParam.srcK], mask, tail, dstRepStride,
            1, reduceParam.srcK / FLOAT_NUM_PER_BLK);
    }
}

__aicore__ inline void ReduceSumLastNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpTensor, const struct ReduceLastND& reduceParam)
{
    const uint32_t splitCount = reduceParam.originalSrcK / FLOAT_REPEAT_SIZE;
    const uint32_t tailSrcK = reduceParam.originalSrcK % FLOAT_REPEAT_SIZE;
    if (splitCount > 0) {
        AlignedReduceSumNDImpl(tmpTensor, src, dst, reduceParam, splitCount);
    }

    if (tailSrcK != 0) {
        ReduceSumLastNDSplitImpl(dst, src, reduceParam, tailSrcK, 1, splitCount);
        PipeBarrier<PIPE_V>();
        if (splitCount == 0) {
            DataCopy(tmpTensor, dst, { 1, (uint16_t)reduceParam.srcM, 0, 0 });
        } else {
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_NUM_PER_BLK);
            Add<float, false>(tmpTensor, tmpTensor, dst, MASK_PLACEHOLDER, 1,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
            SetMaskNorm();
            ResetMask();
        }
    }

    PipeBarrier<PIPE_V>();
    SingleBlockBroadCastImpl(dst, tmpTensor, reduceParam);
}

__aicore__ inline void BasicBlockReduceSumImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t originalSrcM, const uint32_t reduceK)
{
    if (originalSrcM == 1) {
        WholeReduceSum<float, false>(dst, src, MASK_PLACEHOLDER, DEFAULT_REPEAT_STRIDE, 1, 1, 0);
        if (reduceK == DEFAULT_REPEAT_STRIDE * HALF_FACTOR) {
            PipeBarrier<PIPE_V>();
            DataCopy(dst[DEFAULT_REPEAT_STRIDE], dst, { 1, 1, 0, 0 });
        }
    } else {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, originalSrcM * FLOAT_REPEAT_SIZE);
        BlockReduceSum<float, false>(src, src, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(0, originalSrcM * FLOAT_NUM_PER_BLK);
        BlockReduceSum<float, false>(dst, src, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        SetMaskNorm();
        ResetMask();
    }
}

template <bool isBroadCast = true>
__aicore__ inline void NewReduceSumLastNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpTensor, const struct ReduceLastND& reduceParam)
{
    const uint32_t splitCount = reduceParam.originalSrcK / FLOAT_REPEAT_SIZE;
    const uint32_t tailSrcK = reduceParam.originalSrcK % FLOAT_REPEAT_SIZE;
    const uint16_t srcRepeatStride = reduceParam.srcK / FLOAT_NUM_PER_BLK;

    if (reduceParam.originalSrcK < FLOAT_REPEAT_SIZE) { // reduce axis length is (0, 64)
        ReduceSumLastNDSplitImpl(dst, src, reduceParam, reduceParam.originalSrcK, 1, 0);
        ResetMask();
    } else {
        if (reduceParam.originalSrcK >= SOFTMAX_FLOAT_SPECIAL_BLOCKREDUCE_LEN) { // reduce axis length >= 512
            BigBlockReduceSum(tmpTensor, src, splitCount, reduceParam.originalSrcM, reduceParam.srcK);
        } else if (reduceParam.originalSrcK >= HALF_REPEAT_SIZE) { // reduce axis length is [128, 512)
            Add<float, false>(tmpTensor, src, src[FLOAT_REPEAT_SIZE], MASK_PLACEHOLDER,
                (uint8_t)(reduceParam.originalSrcM),
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, (uint8_t)srcRepeatStride, (uint8_t)srcRepeatStride });
            NextBlockAddImpl(tmpTensor, src, (uint8_t)(reduceParam.originalSrcM), srcRepeatStride, splitCount,
                reduceParam.srcK);
        } else { // reduce axis length is [64, 128)
            FirstBlockCopyImpl(tmpTensor, src, reduceParam.originalSrcM, reduceParam.srcK, DEFAULT_REPEAT_STRIDE,
                srcRepeatStride);
        }

        if (tailSrcK != 0) {
            PipeBarrier<PIPE_V>();
            TailAddImpl(tmpTensor, src, reduceParam, tailSrcK, srcRepeatStride, splitCount);
            ResetMask();
        }
        PipeBarrier<PIPE_V>();
        BasicBlockReduceSumImpl(dst, tmpTensor, reduceParam.originalSrcM, reduceParam.dstK);
    }
    if constexpr (isBroadCast) {
        if (reduceParam.originalSrcM != 1 || reduceParam.originalSrcK <= FLOAT_REPEAT_SIZE) {
            PipeBarrier<PIPE_V>();
            AlignedBroadCastImpl(dst, tmpTensor, reduceParam);
        }
    }
}

};
#endif // IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_COMMON_ND_REDUCE_H