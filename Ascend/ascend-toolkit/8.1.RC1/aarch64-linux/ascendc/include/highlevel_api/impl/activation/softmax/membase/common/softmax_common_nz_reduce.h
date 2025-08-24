/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_common_nz_reduce.h 
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_COMMON_NZ_REDUCE_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_COMMON_NZ_REDUCE_H

namespace AscendC {
__aicore__ inline void ReduceMaxSingleBlockNZImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint64_t& mask, const ReduceLastND& reduceParam)
{
    const uint32_t range = reduceParam.srcM / MAX_REPEAT_TIMES;
    const uint32_t tail = reduceParam.srcM % MAX_REPEAT_TIMES;
    for (uint32_t j = 0; j < range; j++) {
        WholeReduceMax(dst[j * MAX_REPEAT_TIMES * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            src[j * MAX_REPEAT_TIMES * SOFTMAX_SHAPE_NZ_BASIC_COUNT], mask, MAX_REPEAT_TIMES, DEFAULT_REPEAT_STRIDE, 1,
            SOFTMAX_SHAPE_NZ_BASIC_COUNT / FLOAT_NUM_PER_BLK);
    }
    if (tail != 0) {
        WholeReduceMax(dst[range * MAX_REPEAT_TIMES * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            src[range * MAX_REPEAT_TIMES * SOFTMAX_SHAPE_NZ_BASIC_COUNT], mask, tail, DEFAULT_REPEAT_STRIDE, 1,
            SOFTMAX_SHAPE_NZ_BASIC_COUNT / FLOAT_NUM_PER_BLK);
    }
}

__aicore__ inline void SingleUnAlignedReduceMaxNZImpl(const LocalTensor<float>& tmpBuffer1,
    const LocalTensor<float>& tmpBuffer0, const uint32_t lastBlockMaskLen, const ReduceLastND& reduceParam)
{
    ReduceMaxSingleBlockNZImpl(tmpBuffer1, tmpBuffer0, lastBlockMaskLen, reduceParam);

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    AlignedColumnBrcbImpl(tmpBuffer1, tmpBuffer1, reduceParam.originalSrcM, SOFTMAX_SHAPE_NZ_BASIC_COUNT);

    ResetMask();
}

__aicore__ inline void ReduceMaxLastNZImpl(const LocalTensor<float>& tmpBuffer1, const LocalTensor<float>& tmpBuffer0,
    uint64_t mask[2], const ReduceLastND& reduceParam)
{
    const uint32_t splitNZBlockCount = reduceParam.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitOffset = reduceParam.dstM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitCount = reduceParam.originalSrcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t lastBlockMaskLen = reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT :
        SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    if (unlikely(splitNZBlockCount == 1 && lastBlockMaskLen != SOFTMAX_SHAPE_NZ_BASIC_COUNT)) {
        SingleUnAlignedReduceMaxNZImpl(tmpBuffer1, tmpBuffer0, lastBlockMaskLen, reduceParam);
    } else {
        if (unlikely(splitNZBlockCount == 1)) {
            ReduceMaxBlockNZImpl(tmpBuffer1, tmpBuffer0, reduceParam);
        } else {
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
            Muls<float, false>(tmpBuffer1, tmpBuffer0, 1.0, MASK_PLACEHOLDER, 1,
                { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 1; j < splitNZBlockCount - 1; j++) {
                Max<float, false>(tmpBuffer1, tmpBuffer1, tmpBuffer0[splitOffset * j], MASK_PLACEHOLDER, 1,
                    { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
                PipeBarrier<PIPE_V>();
            }
            SetMaskNorm();
            ResetMask();

            BinaryComputeWithSpecialMask(tmpBuffer1, tmpBuffer1, tmpBuffer0[splitOffset * (splitNZBlockCount - 1)],
                mask, lastBlockMaskLen, splitCount, Max<float>);

            PipeBarrier<PIPE_V>();
            ReduceMaxBlockNZImpl(tmpBuffer1, tmpBuffer1, reduceParam);
        }

        if (reduceParam.originalSrcM % DEFAULT_REPEAT_STRIDE == 0) {
            PipeBarrier<PIPE_V>();
            BroadCastNZImpl(tmpBuffer1, tmpBuffer1, reduceParam.originalSrcM);
        } else {
            event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);

            ContinusColumnBrcbImpl(tmpBuffer1, tmpBuffer1, reduceParam.originalSrcM, SOFTMAX_SHAPE_NZ_BASIC_COUNT);
            ResetMask();
        }
    }
}
__aicore__ inline void ReduceSumSingleBlockNZImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint64_t& mask, const ReduceLastND& reduceParam)
{
    const uint32_t range = reduceParam.srcM / MAX_REPEAT_TIMES;
    const uint32_t tail = reduceParam.srcM % MAX_REPEAT_TIMES;
    for (uint32_t j = 0; j < range; j++) {
        WholeReduceSum(dst[j * MAX_REPEAT_TIMES * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            src[j * MAX_REPEAT_TIMES * SOFTMAX_SHAPE_NZ_BASIC_COUNT], mask, MAX_REPEAT_TIMES,
            SOFTMAX_SHAPE_NZ_BASIC_COUNT, 1, SOFTMAX_SHAPE_NZ_BASIC_COUNT / FLOAT_NUM_PER_BLK);
    }
    if (tail != 0) {
        WholeReduceSum(dst[range * MAX_REPEAT_TIMES * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            src[range * MAX_REPEAT_TIMES * SOFTMAX_SHAPE_NZ_BASIC_COUNT], mask, tail, SOFTMAX_SHAPE_NZ_BASIC_COUNT, 1,
            SOFTMAX_SHAPE_NZ_BASIC_COUNT / FLOAT_NUM_PER_BLK);
    }
}

__aicore__ inline void SingleUnAlignedReduceSumNZImpl(const LocalTensor<float>& tmpBuffer1,
    const LocalTensor<float>& tmpBuffer0, const uint32_t lastBlockMaskLen, const ReduceLastND& reduceParam)
{
    ReduceSumSingleBlockNZImpl(tmpBuffer1, tmpBuffer0, lastBlockMaskLen, reduceParam);

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    AlignedColumnBrcbImpl(tmpBuffer1, tmpBuffer1, reduceParam.originalSrcM, SOFTMAX_SHAPE_NZ_BASIC_COUNT);

    ResetMask();
}

__aicore__ inline void ReduceSumLastNZImpl(const LocalTensor<float>& tmpBuffer1, const LocalTensor<float>& tmpBuffer0,
    uint64_t mask[2], const struct ReduceLastND& reduceParam)
{
    const uint32_t splitOffset = reduceParam.dstM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitCount = reduceParam.originalSrcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitNZBlockCount = reduceParam.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t lastBlockMaskLen = reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT :
        SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    if (unlikely(splitNZBlockCount == 1 && lastBlockMaskLen != SOFTMAX_SHAPE_NZ_BASIC_COUNT)) {
        SingleUnAlignedReduceSumNZImpl(tmpBuffer1, tmpBuffer0, lastBlockMaskLen, reduceParam);
    } else {
        if (unlikely(splitNZBlockCount == 1)) {
            ReduceSumBlockNZImpl(tmpBuffer1, tmpBuffer0, reduceParam);
        } else {
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
            Muls<float, false>(tmpBuffer1, tmpBuffer0, 1.0, MASK_PLACEHOLDER, 1,
                { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 1; j < splitNZBlockCount - 1; j++) {
                Add<float, false>(tmpBuffer1, tmpBuffer1, tmpBuffer0[splitOffset * j], MASK_PLACEHOLDER, 1,
                    { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
                PipeBarrier<PIPE_V>();
            }
            SetMaskNorm();
            ResetMask();

            BinaryComputeWithSpecialMask(tmpBuffer1, tmpBuffer1, tmpBuffer0[splitOffset * (splitNZBlockCount - 1)],
                mask, lastBlockMaskLen, splitCount, Add<float>);

            PipeBarrier<PIPE_V>();
            ReduceSumBlockNZImpl(tmpBuffer1, tmpBuffer1, reduceParam);
        }

        if (reduceParam.originalSrcM % DEFAULT_REPEAT_STRIDE == 0) {
            PipeBarrier<PIPE_V>();
            BroadCastNZImpl(tmpBuffer1, tmpBuffer1, reduceParam.originalSrcM);
        } else {
            event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);

            ContinusColumnBrcbImpl(tmpBuffer1, tmpBuffer1, reduceParam.originalSrcM, SOFTMAX_SHAPE_NZ_BASIC_COUNT);
            ResetMask();
        }
    }
}

};
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_COMMON_NZ_REDUCE_H