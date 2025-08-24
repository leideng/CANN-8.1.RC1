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
 * \file softmax_common_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_V300_SOFTMAX_COMMON_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_V300_SOFTMAX_COMMON_IMPL_H

namespace AscendC {
__aicore__ inline void BrcbNDImpl(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal,
    const LocalTensor<float>& tmpBuffer, const uint32_t repeat, const uint32_t brcbCount)
{
    Brcb(tmpBuffer, srcLocal, (repeat + BRCB_BROADCAST_NUMBER - 1) / BRCB_BROADCAST_NUMBER,
        { 1, DEFAULT_REPEAT_STRIDE });
    if (brcbCount == DEFAULT_REPEAT_STRIDE) {
        DataCopy(dstLocal, tmpBuffer, { 1, (uint16_t)repeat, 0, 0 });
    } else {
        DataCopy(dstLocal, tmpBuffer, { (uint16_t)repeat, 1, 0, 1 });
        DataCopy(dstLocal[DEFAULT_REPEAT_STRIDE], tmpBuffer, { (uint16_t)repeat, 1, 0, 1 });
    }
}

__aicore__ inline void BrcbNDImpl(const LocalTensor<half>& dstLocal, const LocalTensor<half>& srcLocal,
    const LocalTensor<half>& tmpBuffer, const uint32_t repeat)
{
    Brcb(tmpBuffer, srcLocal, (repeat + BRCB_BROADCAST_NUMBER - 1) / BRCB_BROADCAST_NUMBER,
        { 1, DEFAULT_REPEAT_STRIDE });
    DataCopy(dstLocal, tmpBuffer, { 1, (uint16_t)repeat, 0, 0 });
}

__aicore__ inline void MainBlockMaxImpl(const LocalTensor<half>& dst, const LocalTensor<half>& src,
    const uint8_t splitM, const uint8_t srcRepstride, const uint32_t splitBlock, const uint32_t srcK)
{
    if (splitM > splitBlock) {
        for (uint32_t i = 1; i < splitBlock; ++i) {
            Max(dst, dst, src[HALF_REPEAT_SIZE * i], FULL_MASK_LEN, splitM,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, srcRepstride });
        }
    } else {
        for (uint32_t j = 0; j < splitM; ++j) {
            Max(dst[j * HALF_REPEAT_SIZE], src[HALF_REPEAT_SIZE + j * srcK], dst[j * HALF_REPEAT_SIZE], FULL_MASK_LEN,
                (uint8_t)(splitBlock - 1), { 1, 1, 1, 0, DEFAULT_REPEAT_STRIDE, 0 });
        }
    }
}

__aicore__ inline void MainBlockMaxImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint8_t splitM, const uint8_t srcRepstride, const uint32_t splitBlock, const uint32_t srcK)
{
    if (splitM > splitBlock) {
        for (uint32_t i = 1; i < splitBlock; ++i) {
            Max(dst, dst, src[FLOAT_REPEAT_SIZE * i], MAX_HALF_MASK_LEN, splitM,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, srcRepstride });
        }
    } else {
        for (uint32_t j = 0; j < splitM; ++j) {
            Max(dst[j * FLOAT_REPEAT_SIZE], src[FLOAT_REPEAT_SIZE + j * srcK], dst[j * FLOAT_REPEAT_SIZE],
                MAX_HALF_MASK_LEN, (uint8_t)(splitBlock - 1), { 1, 1, 1, 0, DEFAULT_REPEAT_STRIDE, 0 });
        }
    }
}
__aicore__ inline void MainBlockAddImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint8_t splitM, const uint8_t srcRepstride, const uint32_t splitBlock, const uint32_t srcK)
{
    if (splitM > splitBlock) {
        for (uint32_t i = 1; i < splitBlock; ++i) {
            Add(dst, dst, src[FLOAT_REPEAT_SIZE * i], MAX_HALF_MASK_LEN, splitM,
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, srcRepstride });
        }
    } else {
        for (uint32_t j = 0; j < splitM; ++j) {
            Add(dst[j * FLOAT_REPEAT_SIZE], src[FLOAT_REPEAT_SIZE + j * srcK], dst[j * FLOAT_REPEAT_SIZE],
                MAX_HALF_MASK_LEN, (uint8_t)(splitBlock - 1), { 1, 1, 1, 0, DEFAULT_REPEAT_STRIDE, 0 });
        }
    }
}

__aicore__ inline void ReduceMaxImpl(const LocalTensor<half>& dst, const LocalTensor<half>& src,
    const LocalTensor<half>& tmpTensor, const struct ReduceLastND& reduceParam)
{
    const uint32_t splitCount = reduceParam.originalSrcK / HALF_REPEAT_SIZE;
    const uint32_t tailSrcK = reduceParam.originalSrcK % HALF_REPEAT_SIZE;
    const uint16_t srcRepeatStride = reduceParam.srcK / HALF_NUM_PER_BLK;

    if (reduceParam.originalSrcK <= HALF_REPEAT_SIZE) {
        WholeReduceMax(dst, src, reduceParam.originalSrcK, reduceParam.srcM, 1, 1,
            reduceParam.srcK / HALF_NUM_PER_BLK, ReduceOrder::ORDER_ONLY_VALUE);
    } else {
        DataCopy(tmpTensor, src,
            { (uint16_t)reduceParam.originalSrcM, DEFAULT_REPEAT_STRIDE,
            (uint16_t)((reduceParam.srcK - HALF_REPEAT_SIZE) / HALF_NUM_PER_BLK), 0 });

        MainBlockMaxImpl(tmpTensor, src, (uint8_t)(reduceParam.originalSrcM), srcRepeatStride, splitCount,
            reduceParam.srcK);

        if (tailSrcK != 0) { // mask norm mode
            TailMaxImpl(tmpTensor, src, reduceParam, tailSrcK, srcRepeatStride, splitCount);
        }

        // repeat  = srcM,next need broadcast
        WholeReduceMax(dst, tmpTensor, FULL_MASK_LEN, reduceParam.originalSrcM, 1, 1,
            DEFAULT_REPEAT_STRIDE, ReduceOrder::ORDER_ONLY_VALUE);
    }
    BrcbNDImpl(dst, dst, tmpTensor, reduceParam.originalSrcM);
}

__aicore__ inline void ReduceMaxImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpTensor, const struct ReduceLastND& reduceParam)
{
    const uint32_t splitCount = reduceParam.originalSrcK / FLOAT_REPEAT_SIZE;
    const uint32_t tailSrcK = reduceParam.originalSrcK % FLOAT_REPEAT_SIZE;
    const uint16_t srcRepeatStride = reduceParam.srcK / FLOAT_NUM_PER_BLK;

    if (reduceParam.originalSrcK <= FLOAT_REPEAT_SIZE) {
        WholeReduceMax(dst, src, reduceParam.originalSrcK, reduceParam.srcM, 1, 1,
            reduceParam.srcK / DEFAULT_REPEAT_STRIDE, ReduceOrder::ORDER_ONLY_VALUE);
    } else {
        DataCopy(tmpTensor, src,
            { (uint16_t)reduceParam.originalSrcM, DEFAULT_REPEAT_STRIDE,
            (uint16_t)((reduceParam.srcK - FLOAT_REPEAT_SIZE) / DEFAULT_REPEAT_STRIDE), 0 });

        MainBlockMaxImpl(tmpTensor, src, (uint8_t)(reduceParam.originalSrcM), srcRepeatStride, splitCount,
            reduceParam.srcK);

        if (tailSrcK != 0) { // mask norm mode
            TailMaxImpl(tmpTensor, src, reduceParam, tailSrcK, srcRepeatStride, splitCount);
        }

        // repeat  = srcM,next need broadcast
        WholeReduceMax(dst, tmpTensor, FLOAT_REPEAT_SIZE, reduceParam.originalSrcM, 1, 1,
            DEFAULT_REPEAT_STRIDE, ReduceOrder::ORDER_ONLY_VALUE);
    }
    BrcbNDImpl(dst, dst, tmpTensor, reduceParam.originalSrcM, reduceParam.dstK);
}

__aicore__ inline void ReduceSumImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpTensor, const struct ReduceLastND& reduceParam)
{
    const uint32_t splitCount = reduceParam.originalSrcK / FLOAT_REPEAT_SIZE;
    const uint32_t tailSrcK = reduceParam.originalSrcK % FLOAT_REPEAT_SIZE;
    const uint16_t srcRepeatStride = reduceParam.srcK / FLOAT_NUM_PER_BLK;

    if (reduceParam.originalSrcK <= FLOAT_REPEAT_SIZE) {
        WholeReduceSum(dst, src, reduceParam.originalSrcK, reduceParam.srcM, 1, 1,
            reduceParam.srcK / DEFAULT_REPEAT_STRIDE);
    } else {
        DataCopy(tmpTensor, src,
            { (uint16_t)reduceParam.originalSrcM, DEFAULT_REPEAT_STRIDE,
            (uint16_t)((reduceParam.srcK - FLOAT_REPEAT_SIZE) / DEFAULT_REPEAT_STRIDE), 0 });

        MainBlockAddImpl(tmpTensor, src, (uint8_t)(reduceParam.originalSrcM), srcRepeatStride, splitCount,
            reduceParam.srcK);

        if (tailSrcK != 0) { // mask norm mode
            TailAddImpl(tmpTensor, src, reduceParam, tailSrcK, srcRepeatStride, splitCount);
        }

        // repeat  = srcM,next need broadcast
        WholeReduceSum(dst, tmpTensor, FLOAT_REPEAT_SIZE, reduceParam.originalSrcM, 1, 1,
            DEFAULT_REPEAT_STRIDE);
    }
    BrcbNDImpl(dst, dst, tmpTensor, reduceParam.originalSrcM, reduceParam.dstK);
}

__aicore__ inline void DivNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, const uint32_t originalSrcM, const uint32_t srcK, const uint32_t srcReduceK)
{
    const uint8_t dstStride = srcK / FLOAT_NUM_PER_BLK;
    const uint8_t src1Stride = srcReduceK / FLOAT_NUM_PER_BLK;
    if (srcK < FLOAT_REPEAT_SIZE) {
        const uint16_t repeat = originalSrcM / FLOAT_NUM_PER_BLK;
        const uint16_t tail = originalSrcM % FLOAT_NUM_PER_BLK;

        for (uint8_t j = 0; j < dstStride; j++) {
            Div(dst[j * FLOAT_NUM_PER_BLK], src0[j * FLOAT_NUM_PER_BLK], src1, FLOAT_REPEAT_SIZE, repeat,
                { dstStride, dstStride, src1Stride, (uint8_t)srcK, (uint8_t)srcK, (uint8_t)srcReduceK });
        }
        if (tail != 0) {
            for (uint8_t j = 0; j < dstStride; j++) {
                Div(dst[repeat * FLOAT_NUM_PER_BLK * srcK + j * FLOAT_NUM_PER_BLK],
                    src0[repeat * FLOAT_NUM_PER_BLK * srcK + j * FLOAT_NUM_PER_BLK],
                    src1[repeat * FLOAT_NUM_PER_BLK * srcReduceK], tail * FLOAT_NUM_PER_BLK, 1,
                    { dstStride, dstStride, src1Stride, (uint8_t)srcK, (uint8_t)srcK, (uint8_t)srcReduceK });
            }
        }
    } else if (srcK <= SOFTMAX_MAX_REPEAT_STRIDE && originalSrcM <= MAX_REPEAT_TIMES) {
        const uint32_t range = srcK / FLOAT_REPEAT_SIZE;
        const uint32_t tail = srcK % FLOAT_REPEAT_SIZE;
        for (uint32_t i = 0; i < range; i++) {
            Div(dst[i * FLOAT_REPEAT_SIZE], src0[i * FLOAT_REPEAT_SIZE], src1, FLOAT_REPEAT_SIZE, originalSrcM,
                { 1, 1, 0, (uint8_t)dstStride, (uint8_t)dstStride, src1Stride });
        }
        if (tail != 0) {
            Div(dst[range * FLOAT_REPEAT_SIZE], src0[range * FLOAT_REPEAT_SIZE], src1, tail, originalSrcM,
                { 1, 1, 0, (uint8_t)dstStride, (uint8_t)dstStride, src1Stride });
        }
    } else {
        const uint32_t range = srcK / FLOAT_REPEAT_SIZE;
        const uint32_t tail = srcK % FLOAT_REPEAT_SIZE;
        for (uint32_t j = 0; j < originalSrcM; j++) {
            Div(dst[j * srcK], src0[j * srcK], src1[j * srcReduceK], FLOAT_REPEAT_SIZE, srcK / FLOAT_REPEAT_SIZE,
                { 1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0 });
            if (tail != 0) {
                Div(dst[j * srcK + range * FLOAT_REPEAT_SIZE], src0[j * srcK + range * FLOAT_REPEAT_SIZE],
                    src1[j * srcReduceK], tail, 1, { 1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0 });
            }
        }
    }
}

__aicore__ inline void SubNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, const uint32_t originalSrcM, const uint32_t srcK, const uint32_t srcReduceK)
{
    const uint8_t dstStride = srcK / FLOAT_NUM_PER_BLK;
    const uint8_t src1Stride = srcReduceK / FLOAT_NUM_PER_BLK;
    if (srcK < FLOAT_REPEAT_SIZE) {
        const uint16_t repeat = originalSrcM / FLOAT_NUM_PER_BLK;
        const uint16_t tail = originalSrcM % FLOAT_NUM_PER_BLK;

        for (uint8_t j = 0; j < dstStride; j++) {
            Sub(dst[j * FLOAT_NUM_PER_BLK], src0[j * FLOAT_NUM_PER_BLK], src1, FLOAT_REPEAT_SIZE, repeat,
                { dstStride, dstStride, src1Stride, (uint8_t)srcK, (uint8_t)srcK, (uint8_t)srcReduceK });
        }
        if (tail != 0) {
            for (uint8_t j = 0; j < dstStride; j++) {
                Sub(dst[repeat * FLOAT_NUM_PER_BLK * srcK + j * FLOAT_NUM_PER_BLK],
                    src0[repeat * FLOAT_NUM_PER_BLK * srcK + j * FLOAT_NUM_PER_BLK],
                    src1[repeat * FLOAT_NUM_PER_BLK * srcReduceK], tail * FLOAT_NUM_PER_BLK, 1,
                    { dstStride, dstStride, src1Stride, (uint8_t)srcK, (uint8_t)srcK, (uint8_t)srcReduceK });
            }
        }
    } else if (srcK <= SOFTMAX_MAX_REPEAT_STRIDE && originalSrcM <= MAX_REPEAT_TIMES) {
        const uint32_t range = srcK / FLOAT_REPEAT_SIZE;
        const uint32_t tail = srcK % FLOAT_REPEAT_SIZE;
        for (uint32_t i = 0; i < range; i++) {
            Sub(dst[i * FLOAT_REPEAT_SIZE], src0[i * FLOAT_REPEAT_SIZE], src1, FLOAT_REPEAT_SIZE, originalSrcM,
                { 1, 1, 0, (uint8_t)dstStride, (uint8_t)dstStride, src1Stride });
        }
        if (tail != 0) {
            Sub(dst[range * FLOAT_REPEAT_SIZE], src0[range * FLOAT_REPEAT_SIZE], src1, tail, originalSrcM,
                { 1, 1, 0, (uint8_t)dstStride, (uint8_t)dstStride, src1Stride });
        }
    } else {
        const uint32_t range = srcK / FLOAT_REPEAT_SIZE;
        const uint32_t tail = srcK % FLOAT_REPEAT_SIZE;
        for (uint32_t j = 0; j < originalSrcM; j++) {
            Sub(dst[j * srcK], src0[j * srcK], src1[j * srcReduceK], FLOAT_REPEAT_SIZE, srcK / FLOAT_REPEAT_SIZE,
                { 1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0 });
            if (tail != 0) {
                Sub(dst[j * srcK + range * FLOAT_REPEAT_SIZE], src0[j * srcK + range * FLOAT_REPEAT_SIZE],
                    src1[j * srcReduceK], tail, 1, { 1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0 });
            }
        }
    }
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_V300_SOFTMAX_COMMON_IMPL_H