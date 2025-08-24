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
#ifndef IMPL_ACTIVATION_SOFTMAX_V200_SOFTMAX_COMMON_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_V200_SOFTMAX_COMMON_IMPL_H

#include "../common/softmax_common_nz_reduce.h"

namespace AscendC {
constexpr RoundMode FLOAT2HALF_ROUND_MODE = RoundMode::CAST_NONE;

template <typename T>
__aicore__ inline void BroadCastLastImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const struct BroadCastLastND& brcParam)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    const uint32_t rangeM = brcParam.dstM / SCALAR_STACK_DEPTH;
    const uint32_t tailM = brcParam.dstM % SCALAR_STACK_DEPTH;
    // to compute main block
    for (uint32_t i = 0; i < rangeM; i++) {
        BroadCastLastCompute(dst, src, brcParam, SCALAR_STACK_DEPTH, i);
    }
    // to compute tail M
    BroadCastLastCompute(dst, src, brcParam, tailM, rangeM);
}

template <typename T>
__aicore__ inline void ReduceMaxLastNDSplitImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const struct ReduceLastND& reduceParam, uint64_t mask, uint32_t splitNum)
{
    const uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
    const uint32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    const uint32_t range = reduceParam.srcM / MAX_REPEAT_TIMES;
    const uint32_t tail = reduceParam.srcM % MAX_REPEAT_TIMES;

    for (uint32_t i = 0; i < range; i++) {
        WholeReduceMax(dst[i * MAX_REPEAT_TIMES * reduceParam.dstK],
            src[splitNum * elementNumPerRep + i * MAX_REPEAT_TIMES * reduceParam.srcK], mask, MAX_REPEAT_TIMES,
            reduceParam.dstK / 2, 1, reduceParam.srcK / elementNumPerBlk);
    }
    if (tail != 0) {
        WholeReduceMax(dst[range * MAX_REPEAT_TIMES * reduceParam.dstK],
            src[splitNum * elementNumPerRep + range * MAX_REPEAT_TIMES * reduceParam.srcK], mask, tail,
            reduceParam.dstK / 2, 1, reduceParam.srcK / elementNumPerBlk);
    }
}

template <typename T>
__aicore__ inline void ReduceSumLastNDSplitImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const struct ReduceLastND& reduceParam, uint64_t mask, uint32_t dstRepStride, uint32_t splitNum)
{
    const uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
    const uint32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    const uint32_t range = reduceParam.srcM / MAX_REPEAT_TIMES;
    const uint32_t tail = reduceParam.srcM % MAX_REPEAT_TIMES;

    for (uint32_t i = 0; i < range; i++) {
        WholeReduceSum(dst[i * MAX_REPEAT_TIMES * reduceParam.dstK],
            src[splitNum * elementNumPerRep + i * MAX_REPEAT_TIMES * reduceParam.srcK], mask, MAX_REPEAT_TIMES,
            dstRepStride, 1,
            reduceParam.srcK / elementNumPerBlk);
    }
    if (tail != 0) {
        WholeReduceSum(dst[range * MAX_REPEAT_TIMES * reduceParam.dstK],
            src[splitNum * elementNumPerRep + range * MAX_REPEAT_TIMES * reduceParam.srcK], mask, tail, dstRepStride, 1,
            reduceParam.srcK / elementNumPerBlk);
    }
}

template <typename T>
__aicore__ inline void ReduceSumLastNDImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& tmpTensor, const struct ReduceLastND& reduceParam)
{
    const uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
    const uint32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);

    uint64_t mask = 0;
    uint32_t range = 0;
    uint32_t tail = 0;

    if (reduceParam.originalSrcK <= elementNumPerRep) {
        mask = reduceParam.originalSrcK;
        range = reduceParam.srcM / MAX_REPEAT_TIMES;
        tail = reduceParam.srcM % MAX_REPEAT_TIMES;
        for (uint32_t i = 0; i < range; i++) {
            WholeReduceSum(dst[i * MAX_REPEAT_TIMES * reduceParam.dstK], src[i * MAX_REPEAT_TIMES * reduceParam.srcK],
                mask, MAX_REPEAT_TIMES, reduceParam.dstK, 1,
                reduceParam.srcK / elementNumPerBlk);
        }
        if (tail != 0) {
            WholeReduceSum(dst[range * MAX_REPEAT_TIMES * reduceParam.dstK],
                src[range * MAX_REPEAT_TIMES * reduceParam.srcK], mask, tail, reduceParam.dstK, 1,
                reduceParam.srcK / elementNumPerBlk);
        }
    } else if (reduceParam.originalSrcK > elementNumPerRep) {
        uint32_t splitNum = 0;
        uint32_t offset = reduceParam.dstM * reduceParam.dstK;
        for (splitNum = 0; splitNum < (reduceParam.originalSrcK / elementNumPerRep); splitNum++) {
            mask = elementNumPerRep;
            ReduceSumLastNDSplitImpl(tmpTensor, src, reduceParam, mask, reduceParam.dstK, splitNum);
            PipeBarrier<PIPE_V>();
            if (splitNum == 0) {
                DataCopy(dst, tmpTensor, offset);
            } else {
                Add(dst, dst, tmpTensor, offset);
            }
            PipeBarrier<PIPE_V>();
        }
        if (reduceParam.originalSrcK % elementNumPerRep != 0) {
            mask = reduceParam.originalSrcK % elementNumPerRep;
            ReduceSumLastNDSplitImpl(tmpTensor, src, reduceParam, mask, reduceParam.dstK, splitNum);
            PipeBarrier<PIPE_V>();
            Add(dst, dst, tmpTensor, offset);
        }
    }

    PipeBarrier<PIPE_V>();
    BroadCastLastND brcParam = { reduceParam.dstM, reduceParam.dstK, reduceParam.dstM, reduceParam.dstK };
    BroadCastLastImpl(dst, dst, brcParam);
}

__aicore__ inline void ReduceMaxLastNDImpl(const LocalTensor<float>& dstMax, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpTensor, const struct ReduceLastND& reduceMaxParam)
{
    uint64_t mask = 0;
    uint32_t range = 0;
    uint32_t tail = 0;

    if (reduceMaxParam.originalSrcK <= FLOAT_REPEAT_SIZE) {
        mask = reduceMaxParam.originalSrcK;
        range = reduceMaxParam.originalSrcM / MAX_REPEAT_TIMES;
        tail = reduceMaxParam.originalSrcM % MAX_REPEAT_TIMES;
        for (uint32_t i = 0; i < range; i++) {
            WholeReduceMax(dstMax[i * MAX_REPEAT_TIMES * reduceMaxParam.dstK],
                src[i * MAX_REPEAT_TIMES * reduceMaxParam.srcK], mask, MAX_REPEAT_TIMES,
                reduceMaxParam.dstK / HALF_FACTOR, 1,
                reduceMaxParam.srcK / FLOAT_NUM_PER_BLK);
        }
        if (tail != 0) {
            WholeReduceMax(dstMax[range * MAX_REPEAT_TIMES * reduceMaxParam.dstK],
                src[range * MAX_REPEAT_TIMES * reduceMaxParam.srcK], mask, tail, reduceMaxParam.dstK / HALF_FACTOR, 1,
                reduceMaxParam.srcK / FLOAT_NUM_PER_BLK);
        }
    } else if (reduceMaxParam.originalSrcK > FLOAT_REPEAT_SIZE) {
        mask = FLOAT_REPEAT_SIZE;
        uint32_t splitNum = 0;
        uint32_t offset = reduceMaxParam.dstM * reduceMaxParam.dstK;
        for (splitNum = 0; splitNum < reduceMaxParam.originalSrcK / FLOAT_REPEAT_SIZE; splitNum++) {
            ReduceMaxLastNDSplitImpl(tmpTensor, src, reduceMaxParam, mask, splitNum);
            PipeBarrier<PIPE_V>();
            if (splitNum == 0) {
                DataCopy(dstMax, tmpTensor, offset);
            } else {
                Max(dstMax, dstMax, tmpTensor, offset);
            }
            PipeBarrier<PIPE_V>();
        }
        if (reduceMaxParam.originalSrcK % FLOAT_REPEAT_SIZE != 0) {
            mask = reduceMaxParam.originalSrcK % FLOAT_REPEAT_SIZE;
            ReduceMaxLastNDSplitImpl(tmpTensor, src, reduceMaxParam, mask, splitNum);
            PipeBarrier<PIPE_V>();
            Max(dstMax, dstMax, tmpTensor, offset);
        }
    }
    PipeBarrier<PIPE_V>();
    BroadCastLastND brcParam = { reduceMaxParam.dstM, reduceMaxParam.dstK, reduceMaxParam.dstM, reduceMaxParam.dstK };
    BroadCastLastImpl(dstMax, dstMax, brcParam);
}
__aicore__ inline void FirstBlockCopyImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t srcM, const uint32_t srcK, const uint16_t dstRepeatStride, const uint16_t srcRepeatStride)
{
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, srcM * FLOAT_REPEAT_SIZE);
    Adds<float, false>(dst, src, 0, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, (uint8_t)(srcK / DEFAULT_REPEAT_STRIDE) });
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void BroadCastImpl(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal,
    const uint32_t& repeat, const uint32_t& brcbCount)
{
    float scalarList[SCALAR_STACK_DEPTH] = {0};
    SetVectorMask<float, MaskMode::NORMAL>(brcbCount);
    const uint32_t rangeM = repeat / SCALAR_STACK_DEPTH;
    const uint32_t tailM = repeat % SCALAR_STACK_DEPTH;

    for (uint32_t i = 0; i < rangeM; i++) {
        for (uint32_t j = 0; j < SCALAR_STACK_DEPTH; j++) {
            scalarList[j] = srcLocal.GetValue(i * SCALAR_STACK_DEPTH + j);
        }
        for (uint32_t k = 0; k < SCALAR_STACK_DEPTH; k++) {
            Duplicate<float, false>(dstLocal[i * brcbCount * SCALAR_STACK_DEPTH + k * brcbCount], scalarList[k],
                MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        }
    }
    if (tailM != 0) {
        for (uint32_t j = 0; j < tailM; j++) {
            scalarList[j] = srcLocal.GetValue(rangeM * SCALAR_STACK_DEPTH + j);
        }
        for (uint32_t k = 0; k < tailM; k++) {
            Duplicate<float, false>(dstLocal[rangeM * brcbCount * SCALAR_STACK_DEPTH + k * brcbCount], scalarList[k],
                MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        }
    }
}
__aicore__ inline void BroadCastNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t originalSrcM)
{
    uint8_t repeat = DivCeil(originalSrcM, DEFAULT_REPEAT_STRIDE);
    for (uint8_t i = 0; i < repeat; i++) {
        Muls<float, false>(dst[i * B16_BYTE_SIZE * FLOAT_REPEAT_SIZE], src[i * DEFAULT_REPEAT_STRIDE], 1.0,
            MASK_PLACEHOLDER, B16_BYTE_SIZE, { 1, 0, DEFAULT_REPEAT_STRIDE, 0 });
    }
    PipeBarrier<PIPE_V>();

    uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE];
    for (int32_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstList[i] = (uint64_t)dst[i * FLOAT_NUM_PER_BLK].GetPhyAddr();
        srcList[i] = (uint64_t)dst[i * FLOAT_NUM_PER_BLK].GetPhyAddr();
    }

    TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = repeat;
    if (transDataParams.repeatTimes > 1) {
        transDataParams.dstRepStride = B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE;
        transDataParams.srcRepStride = B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE;
    }
    TransDataTo5HD<float>(dstList, srcList, transDataParams);
}

__aicore__ inline void BasicBlockReduceMaxImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t originalSrcM, const uint32_t reduceK)
{
    if (originalSrcM == 1) {
        WholeReduceMax<float, false>(dst, src, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float scalarVal = dst.GetValue(0);
        Duplicate(dst, scalarVal, reduceK, 1, 1, DEFAULT_REPEAT_STRIDE);
        ResetMask();
    } else {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, originalSrcM * FLOAT_REPEAT_SIZE);
        BlockReduceMax<float, false>(src, src, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(0, originalSrcM * FLOAT_NUM_PER_BLK);
        BlockReduceMax<float, false>(dst, src, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        SetMaskNorm();
        ResetMask();

        PipeBarrier<PIPE_V>();
        BroadCastNDImpl(src, dst, originalSrcM);
        PipeBarrier<PIPE_V>();
        if (reduceK == FLOAT_NUM_PER_BLK) {
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(0, originalSrcM * FLOAT_NUM_PER_BLK);
            Adds<float, false>(dst, src, 0, MASK_PLACEHOLDER, 1,
                { 1, HALF_FACTOR, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE * HALF_FACTOR});
            SetMaskNorm();
            ResetMask();
        } else {
            Adds<float>(dst, src, 0, originalSrcM * reduceK);
        }
    }
}
__aicore__ inline void NewReduceMaxLastNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpTensor, const struct ReduceLastND& reduceParam)
{
    ResetMask();
    const uint32_t splitCount = reduceParam.originalSrcK / FLOAT_REPEAT_SIZE;
    const uint32_t tailSrcK = reduceParam.originalSrcK % FLOAT_REPEAT_SIZE;
    const uint16_t srcRepeatStride = reduceParam.srcK / FLOAT_NUM_PER_BLK;

    if (reduceParam.originalSrcK < FLOAT_REPEAT_SIZE) {
        ReduceMaxLastNDSplitImpl(dst, src, reduceParam, reduceParam.originalSrcK, 0);
        PipeBarrier<PIPE_V>();
        BroadCastLastND brcParam = { reduceParam.dstM, reduceParam.dstK, reduceParam.dstM, reduceParam.dstK };
        BroadCastLastImpl(dst, dst, brcParam);
    } else {
        if (reduceParam.originalSrcK >= SOFTMAX_FLOAT_SPECIAL_BLOCKREDUCE_LEN) {
            BigBlockReduceMax(tmpTensor, src, splitCount, reduceParam.originalSrcM, reduceParam.srcK);
        } else if (reduceParam.originalSrcK >= HALF_REPEAT_SIZE) {
            Max<float, false>(tmpTensor, src, src[FLOAT_REPEAT_SIZE], MASK_PLACEHOLDER,
                (uint8_t)(reduceParam.originalSrcM),
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, (uint8_t)srcRepeatStride, (uint8_t)srcRepeatStride });
            NextBlockMaxImpl(tmpTensor, src, (uint8_t)(reduceParam.originalSrcM), srcRepeatStride, splitCount,
                reduceParam.srcK);
        } else {
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
}

__aicore__ inline void BasicBlockReduceAddImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t originalSrcM, const uint32_t reduceK)
{
    if (originalSrcM == 1) {
        WholeReduceSum<float, false>(dst, src, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float scalarVal = dst.GetValue(0);
        Duplicate(dst, scalarVal, reduceK, 1, 1, DEFAULT_REPEAT_STRIDE);
        ResetMask();
    } else {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, originalSrcM * FLOAT_REPEAT_SIZE);
        BlockReduceSum<float, false>(src, src, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(0, originalSrcM * FLOAT_NUM_PER_BLK);
        BlockReduceSum<float, false>(dst, src, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
        SetMaskNorm();
        ResetMask();

        PipeBarrier<PIPE_V>();
        BroadCastNDImpl(src, dst, originalSrcM);
        PipeBarrier<PIPE_V>();
        if (reduceK == FLOAT_NUM_PER_BLK) {
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(0, originalSrcM * FLOAT_NUM_PER_BLK);
            Adds<float, false>(dst, src, 0, MASK_PLACEHOLDER, 1,
                { 1, HALF_FACTOR, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE * HALF_FACTOR});
            SetMaskNorm();
            ResetMask();
        } else {
            Adds<float>(dst, src, 0, originalSrcM * reduceK);
        }
    }
}
__aicore__ inline void NewReduceSumLastNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpTensor, const struct ReduceLastND& reduceParam)
{
    ResetMask();
    const uint32_t splitCount = reduceParam.originalSrcK / FLOAT_REPEAT_SIZE;
    const uint32_t tailSrcK = reduceParam.originalSrcK % FLOAT_REPEAT_SIZE;
    const uint16_t srcRepeatStride = reduceParam.srcK / FLOAT_NUM_PER_BLK;

    if (reduceParam.originalSrcK < FLOAT_REPEAT_SIZE) {
        ReduceSumLastNDSplitImpl(dst, src, reduceParam, reduceParam.originalSrcK, reduceParam.dstK, 0);
        PipeBarrier<PIPE_V>();
        BroadCastLastND brcParam = { reduceParam.dstM, reduceParam.dstK, reduceParam.dstM, reduceParam.dstK };
        BroadCastLastImpl(dst, dst, brcParam);
    } else {
        if (reduceParam.originalSrcK >= SOFTMAX_FLOAT_SPECIAL_BLOCKREDUCE_LEN) {
            BigBlockReduceSum(tmpTensor, src, splitCount, reduceParam.originalSrcM, reduceParam.srcK);
        } else if (reduceParam.originalSrcK >= HALF_REPEAT_SIZE) {
            Add<float, false>(tmpTensor, src, src[FLOAT_REPEAT_SIZE], MASK_PLACEHOLDER,
                (uint8_t)(reduceParam.originalSrcM),
                { 1, 1, 1, DEFAULT_REPEAT_STRIDE, (uint8_t)srcRepeatStride, (uint8_t)srcRepeatStride });
            NextBlockAddImpl(tmpTensor, src, (uint8_t)(reduceParam.originalSrcM), srcRepeatStride, splitCount,
                reduceParam.srcK);
        } else {
            FirstBlockCopyImpl(tmpTensor, src, reduceParam.originalSrcM, reduceParam.srcK, DEFAULT_REPEAT_STRIDE,
                srcRepeatStride);
        }

        if (tailSrcK != 0) {
            PipeBarrier<PIPE_V>();
            TailAddImpl(tmpTensor, src, reduceParam, tailSrcK, srcRepeatStride, splitCount);
            ResetMask();
        }
        PipeBarrier<PIPE_V>();
        BasicBlockReduceAddImpl(dst, tmpTensor, reduceParam.originalSrcM, reduceParam.dstK);
    }
}
};
#endif // IMPL_ACTIVATION_SOFTMAX_V200_SOFTMAX_COMMON_IMPL_H
