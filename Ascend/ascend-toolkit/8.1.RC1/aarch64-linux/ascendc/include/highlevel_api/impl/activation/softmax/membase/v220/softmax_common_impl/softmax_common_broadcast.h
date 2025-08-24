/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_common_broadcast.h 
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_COMMON_BROADCAST_H
#define IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_COMMON_BROADCAST_H 

namespace AscendC {

template <typename T>
__aicore__ inline void BroadCastLastImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const struct BroadCastLastND& brcParam)
{
#if ASCENDC_CPU_DEBUG == 0
    // enable mask count mode
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T); // half is 16, float is 8
    uint64_t lowMask =
        brcParam.srcM * elementNumPerBlk; // MASK[31:0] is the element count to be processed per repeat time
    uint16_t repeat = brcParam.dstK / elementNumPerBlk;
    uint16_t srcBlkStride = brcParam.srcK / elementNumPerBlk; // 64B is 2,32B is 1
    uint64_t mask[2] = { lowMask, 0 };

    uint32_t range = repeat / MAX_REPEAT_TIMES;
    uint32_t tail = repeat % MAX_REPEAT_TIMES;

    SetMaskCount();
    for (uint32_t i = 0; i < range; i++) {
        Copy<T>(dst[i * elementNumPerBlk * MAX_REPEAT_TIMES], src, mask, MAX_REPEAT_TIMES,
            { repeat, srcBlkStride, 1, 0 });
    }
    if (tail != 0) {
        Copy<T>(dst[range * elementNumPerBlk * MAX_REPEAT_TIMES], src, mask, tail, { repeat, srcBlkStride, 1, 0 });
    }

    SetMaskNorm();
    ResetMask();
#else
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
#endif
}

__aicore__ inline void SingleBlockBroadCastImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const struct ReduceLastND& reduceParam)
{
    BrcbRepeatParams brcbParams;
    brcbParams.dstBlkStride = 1;
    brcbParams.dstRepStride = BRCB_BROADCAST_NUMBER;
    const uint32_t range = reduceParam.originalSrcM / BRCB_BROADCAST_NUMBER;
    const uint32_t tail = reduceParam.originalSrcM % BRCB_BROADCAST_NUMBER;

    if (range != 0) {
        if (reduceParam.dstK == BRCB_BROADCAST_NUMBER * HALF_FACTOR) { // when src is float type and reduce.dst = 64B
            brcbParams.dstBlkStride = HALF_FACTOR;
            brcbParams.dstRepStride = BRCB_BROADCAST_NUMBER * HALF_FACTOR;
            Brcb(dst[0], src, range, brcbParams);
            Brcb(dst[BRCB_BROADCAST_NUMBER], src, range, brcbParams);
        } else {
            Brcb(dst, src, range, brcbParams);
        }
    }

    if (tail != 0) { // use duplicate in tail
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float scalarList[SCALAR_STACK_DEPTH] = {0};
        for (uint32_t j = 0; j < tail; j++) {
            scalarList[j] = src[(range * BRCB_BROADCAST_NUMBER + j)].GetValue(0);
        }

        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        for (uint32_t k = 0; k < tail; k++) {
            Duplicate(dst[(range * SCALAR_STACK_DEPTH + k) * reduceParam.dstK], scalarList[k], reduceParam.dstK, 1,
                DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        }
    }
}

__aicore__ inline void AlignedBroadCastImpl(const LocalTensor<float>& dst, const LocalTensor<float>& tmpbuffer,
    const struct ReduceLastND& reduceParam)
{
    const uint32_t repeat = (reduceParam.originalSrcM + BRCB_BROADCAST_NUMBER - 1) / BRCB_BROADCAST_NUMBER;

    if (reduceParam.dstK == BRCB_BROADCAST_NUMBER * HALF_FACTOR) { // when src is float type and reduce.dst = 64B
        if (reduceParam.originalSrcM != 1) {
            Brcb(tmpbuffer, dst, (uint8_t)repeat, { HALF_FACTOR, BRCB_BROADCAST_NUMBER * HALF_FACTOR });
            Brcb(tmpbuffer[BRCB_BROADCAST_NUMBER], dst, (uint8_t)repeat,
                { HALF_FACTOR, BRCB_BROADCAST_NUMBER * HALF_FACTOR });
        } else {
            Brcb(tmpbuffer, dst, (uint8_t)repeat, { 1, BRCB_BROADCAST_NUMBER });
            PipeBarrier<PIPE_V>();
            DataCopy(tmpbuffer[DEFAULT_REPEAT_STRIDE], tmpbuffer, { 1, 1, 0, 0 });
        }
    } else {
        Brcb(tmpbuffer, dst, (uint8_t)repeat, { 1, BRCB_BROADCAST_NUMBER });
    }
    PipeBarrier<PIPE_V>();
    // copy to dst
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.originalSrcM * reduceParam.dstK);
    Copy<float, false>(dst, tmpbuffer, MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    SetMaskNorm();
    ResetMask();
}

};
#endif // IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_COMMON_BROADCAST_H