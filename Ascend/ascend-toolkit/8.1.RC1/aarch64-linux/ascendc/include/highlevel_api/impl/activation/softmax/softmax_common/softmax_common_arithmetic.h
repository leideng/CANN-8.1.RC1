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
 * \file softmax_common_arithmetic.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_COMMON_ARITHMETIC_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_COMMON_ARITHMETIC_H
#include "softmax_common_utils.h"

namespace AscendC {

__aicore__ inline void TailMaxImpl(const LocalTensor<half>& dst, const LocalTensor<half>& src,
    const ReduceLastND& reduceParam, const uint64_t mask, const uint8_t srcRepeatStride, const uint32_t splitCount)
{
    const uint32_t tailStartOffset = HALF_REPEAT_SIZE * splitCount;
    if (reduceParam.srcK > SOFTMAX_MAX_REPEAT_STRIDE) { // repstride support max 255
        for (uint32_t i = 0; i < reduceParam.originalSrcM; i++) {
            Max(dst[i * HALF_REPEAT_SIZE], dst[i * HALF_REPEAT_SIZE], src[tailStartOffset + i * reduceParam.srcK], mask,
                1, { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        }
    } else {
        const uint32_t range = reduceParam.originalSrcM / MAX_REPEAT_TIMES;
        const uint32_t tail = reduceParam.originalSrcM % MAX_REPEAT_TIMES;
        for (uint32_t i = 0; i < range; i++) {
            Max(dst[i * SOFTMAX_MAX_REPEAT_CLC_HALF_NUM], dst[i * SOFTMAX_MAX_REPEAT_CLC_HALF_NUM],
                src[tailStartOffset + i * MAX_REPEAT_TIMES * reduceParam.srcK], mask, MAX_REPEAT_TIMES,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, srcRepeatStride });
        }
        if (tail != 0) {
            Max(dst[range * SOFTMAX_MAX_REPEAT_CLC_HALF_NUM], dst[range * SOFTMAX_MAX_REPEAT_CLC_HALF_NUM],
                src[tailStartOffset + range * MAX_REPEAT_TIMES * reduceParam.srcK], mask, tail,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, srcRepeatStride });
        }
    }
}
__aicore__ inline void TailMaxImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const ReduceLastND& reduceParam, const uint64_t mask, const uint8_t srcRepeatStride, const uint32_t splitCount)
{
    const uint32_t tailStartOffset = FLOAT_REPEAT_SIZE * splitCount;
    if (reduceParam.srcK > SOFTMAX_MAX_REPEAT_STRIDE) { // repstride support max 255
        for (uint32_t i = 0; i < reduceParam.originalSrcM; i++) {
            Max(dst[i * FLOAT_REPEAT_SIZE], dst[i * FLOAT_REPEAT_SIZE], src[tailStartOffset + i * reduceParam.srcK],
                mask, 1, { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        }
    } else {
        const uint32_t range = reduceParam.originalSrcM / MAX_REPEAT_TIMES;
        const uint32_t tail = reduceParam.originalSrcM % MAX_REPEAT_TIMES;
        for (uint32_t i = 0; i < range; i++) {
            Max(dst[i * SOFTMAX_MAX_REPEAT_CLC_FLOAT_NUM], dst[i * SOFTMAX_MAX_REPEAT_CLC_FLOAT_NUM],
                src[tailStartOffset + i * MAX_REPEAT_TIMES * reduceParam.srcK], mask, MAX_REPEAT_TIMES,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, srcRepeatStride });
        }
        if (tail != 0) {
            Max(dst[range * SOFTMAX_MAX_REPEAT_CLC_FLOAT_NUM], dst[range * SOFTMAX_MAX_REPEAT_CLC_FLOAT_NUM],
                src[tailStartOffset + range * MAX_REPEAT_TIMES * reduceParam.srcK], mask, tail,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, srcRepeatStride });
        }
    }
}
__aicore__ inline void TailAddImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const ReduceLastND& reduceParam, const uint64_t mask, const uint8_t srcRepeatStride, const uint32_t splitCount)
{
    const uint32_t tailStartOffset = FLOAT_REPEAT_SIZE * splitCount;
    if (reduceParam.srcK > SOFTMAX_MAX_REPEAT_STRIDE) { // repstride support max 255
        for (uint32_t i = 0; i < reduceParam.originalSrcM; i++) {
            Add(dst[i * FLOAT_REPEAT_SIZE], dst[i * FLOAT_REPEAT_SIZE], src[tailStartOffset + i * reduceParam.srcK],
                mask, 1, { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        }
    } else {
        const uint32_t range = reduceParam.originalSrcM / MAX_REPEAT_TIMES;
        const uint32_t tail = reduceParam.originalSrcM % MAX_REPEAT_TIMES;
        for (uint32_t i = 0; i < range; i++) {
            Add(dst[i * SOFTMAX_MAX_REPEAT_CLC_FLOAT_NUM], dst[i * SOFTMAX_MAX_REPEAT_CLC_FLOAT_NUM],
                src[tailStartOffset + i * MAX_REPEAT_TIMES * reduceParam.srcK], mask, MAX_REPEAT_TIMES,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, srcRepeatStride });
        }
        if (tail != 0) {
            Add(dst[range * SOFTMAX_MAX_REPEAT_CLC_FLOAT_NUM], dst[range * SOFTMAX_MAX_REPEAT_CLC_FLOAT_NUM],
                src[tailStartOffset + range * MAX_REPEAT_TIMES * reduceParam.srcK], mask, tail,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, srcRepeatStride });
        }
    }
}

__aicore__ inline void NextBlockMaxImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint8_t splitM, const uint8_t srcRepstride, const uint32_t splitBlock, const uint32_t srcK)
{
    if (splitM > splitBlock) {
        for (uint32_t i = HALF_FACTOR; i < splitBlock; ++i) {
            PipeBarrier<PIPE_V>();
            Max<float, false>(dst, dst, src[FLOAT_REPEAT_SIZE * i], 1, splitM,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, srcRepstride });
        }
    } else {
#ifdef ASCENDC_CPU_DEBUG
        if (splitBlock == HALF_FACTOR) {
            return;
        }
#endif
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitM; ++j) {
            Max<float, false>(dst[j * FLOAT_REPEAT_SIZE], src[HALF_REPEAT_SIZE + j * srcK], dst[j * FLOAT_REPEAT_SIZE],
                1, (uint8_t)(splitBlock - HALF_FACTOR), { 1, 1, 1, 0, DEFAULT_REPEAT_STRIDE, 0 });
        }
    }
}

__aicore__ inline void NextBlockAddImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint8_t splitM, const uint8_t srcRepstride, const uint32_t splitBlock, const uint32_t srcK)
{
    if (splitM > splitBlock) {
        for (uint32_t i = HALF_FACTOR; i < splitBlock; ++i) {
            PipeBarrier<PIPE_V>();
            Add<float, false>(dst, dst, src[FLOAT_REPEAT_SIZE * i], 1, splitM,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, srcRepstride });
        }
    } else {
#ifdef ASCENDC_CPU_DEBUG
        if (splitBlock == HALF_FACTOR) {
            return;
        }
#endif
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitM; ++j) {
            Add<float, false>(dst[j * FLOAT_REPEAT_SIZE], src[HALF_REPEAT_SIZE + j * srcK], dst[j * FLOAT_REPEAT_SIZE],
                1, (uint8_t)(splitBlock - HALF_FACTOR), { 1, 1, 1, 0, DEFAULT_REPEAT_STRIDE, 0 });
        }
    }
}

__aicore__ inline void BasicBlockMaxImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint8_t splitM,
    uint8_t offset, const uint32_t splitBlock)
{
    Max<float, false>(dst, src, src[FLOAT_REPEAT_SIZE], 1, splitM, { 1, 1, 1, DEFAULT_REPEAT_STRIDE, offset, offset });
    if (splitM > splitBlock) {
        for (uint32_t i = 2; i < splitBlock; ++i) {
            PipeBarrier<PIPE_V>();
            Max<float, false>(dst, dst, src[FLOAT_REPEAT_SIZE * i], 1, splitM,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, offset });
        }
    } else {
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitM; ++j) {
            Max<float, false>(dst[j * FLOAT_REPEAT_SIZE],
                src[HALF_FACTOR * FLOAT_REPEAT_SIZE + splitBlock * FLOAT_REPEAT_SIZE * j], dst[j * FLOAT_REPEAT_SIZE],
                1, (uint8_t)(splitBlock - HALF_FACTOR), { 1, 1, 1, 0, DEFAULT_REPEAT_STRIDE, 0 });
        }
    }
}

__aicore__ inline void BasicBlockAddImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint8_t splitM,
    uint8_t offset, const uint32_t splitBlock)
{
    Add<float, false>(dst, src, src[FLOAT_REPEAT_SIZE], 1, splitM, { 1, 1, 1, DEFAULT_REPEAT_STRIDE, offset, offset });
    if (splitM > splitBlock) {
        for (uint32_t i = 2; i < splitBlock; ++i) {
            PipeBarrier<PIPE_V>();
            Add<float, false>(dst, dst, src[FLOAT_REPEAT_SIZE * i], 1, splitM,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, offset });
        }
    } else {
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitM; ++j) {
            Add<float, false>(dst[j * FLOAT_REPEAT_SIZE],
                src[HALF_FACTOR * FLOAT_REPEAT_SIZE + splitBlock * FLOAT_REPEAT_SIZE * j], dst[j * FLOAT_REPEAT_SIZE],
                1, (uint8_t)(splitBlock - HALF_FACTOR), { 1, 1, 1, 0, DEFAULT_REPEAT_STRIDE, 0 });
        }
    }
}

__aicore__ inline void GenericSubNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, const uint32_t originalSrcM, const uint32_t srcK, const uint32_t srcReduceK)
{
    if (srcK < SOFTMAX_SUB_DIV_ROW_COLUMN_SIZE) {
        const uint8_t dstBlockStride = srcK / FLOAT_NUM_PER_BLK;
        const uint8_t src1BlockStride = srcReduceK / FLOAT_NUM_PER_BLK;
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, originalSrcM * FLOAT_NUM_PER_BLK);
        for (uint8_t j = 0; j < dstBlockStride; j++) {
            Sub<float, false>(dst[j * FLOAT_NUM_PER_BLK], src0[j * FLOAT_NUM_PER_BLK], src1, 1, 1,
                { dstBlockStride, dstBlockStride, src1BlockStride, (uint8_t)srcK, (uint8_t)srcK, (uint8_t)srcReduceK });
        }
        SetMaskNorm();
        ResetMask();
    } else {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, srcK);
        for (uint32_t j = 0; j < originalSrcM; j++) {
            Sub<float, false>(dst[j * srcK], src0[j * srcK], src1[j * srcReduceK], 1, 1,
                { 1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0 });
        }
        SetMaskNorm();
        ResetMask();
    }
}

__aicore__ inline void GenericDivNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, const uint32_t originalSrcM, const uint32_t srcK, const uint32_t srcReduceK)
{
    if (srcK < SOFTMAX_SUB_DIV_ROW_COLUMN_SIZE) {
        const uint8_t dstBlockStride = srcK / FLOAT_NUM_PER_BLK;
        const uint8_t src1BlockStride = srcReduceK / FLOAT_NUM_PER_BLK;
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, originalSrcM * FLOAT_NUM_PER_BLK);
        for (uint8_t j = 0; j < dstBlockStride; j++) {
            Div<float, false>(dst[j * FLOAT_NUM_PER_BLK], src0[j * FLOAT_NUM_PER_BLK], src1, 1, 1,
                { dstBlockStride, dstBlockStride, src1BlockStride, (uint8_t)srcK, (uint8_t)srcK, (uint8_t)srcReduceK });
        }
        SetMaskNorm();
        ResetMask();
    } else {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, srcK);
        for (uint32_t j = 0; j < originalSrcM; j++) {
            Div<float, false>(dst[j * srcK], src0[j * srcK], src1[j * srcReduceK], 1, 1,
                { 1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0 });
        }
        SetMaskNorm();
        ResetMask();
    }
}

__aicore__ inline void GenericMulNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, const uint32_t originalSrcM, const uint32_t srcK, const uint32_t srcReduceK)
{
    if (srcK < SOFTMAX_SUB_DIV_ROW_COLUMN_SIZE) {
        const uint8_t dstBlockStride = srcK / FLOAT_NUM_PER_BLK;
        const uint8_t src1BlockStride = srcReduceK / FLOAT_NUM_PER_BLK;
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, originalSrcM * FLOAT_NUM_PER_BLK);
        for (uint8_t j = 0; j < dstBlockStride; j++) {
            Mul<float, false>(dst[j * FLOAT_NUM_PER_BLK], src0[j * FLOAT_NUM_PER_BLK], src1, 1, 1,
                { dstBlockStride, dstBlockStride, src1BlockStride, (uint8_t)srcK, (uint8_t)srcK, (uint8_t)srcReduceK });
        }
        SetMaskNorm();
        ResetMask();
    } else {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, srcK);
        for (uint32_t j = 0; j < originalSrcM; j++) {
            Mul<float, false>(dst[j * srcK], src0[j * srcK], src1[j * srcReduceK], 1, 1,
                { 1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0 });
        }
        SetMaskNorm();
        ResetMask();
    }
}

__aicore__ inline void TransDivToMulImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpbuffer, const uint32_t originalSrcM, const uint32_t srcK, const uint32_t srcReduceK)
{
    const uint32_t curReduceSize = originalSrcM * srcReduceK;
    Duplicate(tmpbuffer, (float)1.0, curReduceSize);
    PipeBarrier<PIPE_V>();
    Div(tmpbuffer, tmpbuffer, src, curReduceSize);
    PipeBarrier<PIPE_V>();
    GenericMulNDImpl(dst, dst, tmpbuffer, originalSrcM, srcK, srcReduceK);
}

}; // namespace AscendC
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_COMMON_ARITHMETIC_H