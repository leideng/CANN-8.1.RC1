/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file kernel_operator_group_barrier_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_GROUP_BARRIER_IMPL_H
#define ASCENDC_MODULE_OPERATOR_GROUP_BARRIER_IMPL_H
#include "core_mng/roc/kernel_operator_group_barrier_intf.h"
namespace AscendC {
__aicore__ inline int64_t GetBlockNum();
template <PipeMode pipeMode>
__aicore__ inline GroupBarrier<pipeMode>::GroupBarrier(
    GM_ADDR groupWorkspace, uint32_t arriveSizeIn, uint32_t waitSizeIn)
{
    if ASCEND_IS_AIV {
        ASCENDC_DEBUG_ASSERT(
            (pipeMode == PipeMode::MTE3_MODE), "Currently GroupBarrier only support PipeMode::MTE3_MODE");
        ASCENDC_DEBUG_ASSERT((arriveSizeIn > 0), "arriveSizeIn is %u, which should be larger than 0", arriveSizeIn);
        ASCENDC_DEBUG_ASSERT((waitSizeIn > 0), "waitSizeIn is %u, which should be larger than 0", waitSizeIn);
        ASCENDC_DEBUG_ASSERT((waitSizeIn <= GetBlockNum()),
            "waitSizeIn %u is larger than max waitSize is %lld \n",
            waitSizeIn,
            GetBlockNum());
        ASCENDC_DEBUG_ASSERT((arriveSizeIn <= GetBlockNum()),
            "waitSize is %u is larger than max arriveSize is %lld \n",
            arriveSizeIn,
            GetBlockNum());
        this->barrierInfoArrive = reinterpret_cast<__gm__ BarrierInfo *>(groupWorkspace);
        this->barrierInfoWait = reinterpret_cast<__gm__ BarrierInfo *>(groupWorkspace + BARRIER_SIZE);
        this->arriveSize = arriveSizeIn;
        this->waitSize = waitSizeIn;
        this->counter = 1;
        this->hasArrive = false;

        // worst case: max aiv wait max aiv, thus need at least max aiv block which first element is 1 to atomic add
#if ASCENDC_CPU_DEBUG
        __ubuf__ int32_t *dst = reinterpret_cast<__ubuf__ int32_t *>(
            GetTPipePtr()->GetBaseAddr((uint8_t)(TPosition::VECOUT)) + UB_START_ADDR);
#else
        __ubuf__ int32_t *dst = reinterpret_cast<__ubuf__ int32_t *>(UB_START_ADDR);
#endif
        for (uint32_t i = 0; i < BARRIER_MAX_AIV; i++) {
            *(__ubuf__ uint32_t *)(dst + DEFAULT_BLK_NUM * i) = 1;  // Set first element in each block to be 1
        }
    }
}

template <PipeMode pipeMode>
__aicore__ inline void GroupBarrier<pipeMode>::Arrive(uint32_t arriveIndex)
{
    if ASCEND_IS_AIV {
        if (counter > 1) {  // must wait last round to end
            uint32_t expectedWaitNum = (counter - 1) * waitSize;
            GlobalTensor<uint32_t> barrierInfoWaitGlobal;
            __gm__ BarrierInfo *BarrierInfoAddr =
                barrierInfoWait + (CACHE_LINE_LEN / sizeof(BarrierInfo)) * arriveIndex;
            dcci((__gm__ uint64_t *)BarrierInfoAddr, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
            while (BarrierInfoAddr->head != expectedWaitNum) {  // check wait in last round all finished
                dcci((__gm__ uint64_t *)BarrierInfoAddr, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
            }
        }
        __WriteCurrentValue(barrierInfoArrive);
        counter += 1;
        hasArrive = true;
    }
}

// stuck in while loop until all aiv has arrived, then update wait counter
template <PipeMode pipeMode>
__aicore__ inline void GroupBarrier<pipeMode>::Wait(uint32_t waitIndex)
{
    // Get the counter by whether that aiv call arrive before wait
    // Ex: aiv call arrive + wait: In arrive, counter++. thus in wait, should counter - 1
    // Ex: aiv call only wait:     No arrive. thus no need to update counter
    if ASCEND_IS_AIV {
        uint32_t waitCounter = (hasArrive) ? counter - 1 : counter;
        uint32_t expectedArriveNum = waitCounter * arriveSize;
        __gm__ BarrierInfo *BarrierInfoAddr = barrierInfoArrive + (CACHE_LINE_LEN / sizeof(BarrierInfo)) * waitIndex;
        dcci((__gm__ uint64_t *)BarrierInfoAddr, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        while (BarrierInfoAddr->head < expectedArriveNum) {  // check in current round, all aiv has arrived
            dcci((__gm__ uint64_t *)BarrierInfoAddr, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        }
        __WriteCurrentValue(barrierInfoWait);
        counter =
            (hasArrive) ? counter : counter + 1;  // If counter updated by calling arrive before, no need to update
        hasArrive = false;
    }
}

template <PipeMode pipeMode>
__aicore__ inline uint64_t GroupBarrier<pipeMode>::GetWorkspaceLen()
{
    if ASCEND_IS_AIV {
        ASCENDC_DEBUG_ASSERT((arriveSize > 0), "arriveSize is %u, it must be larger than 0", arriveSize);
        ASCENDC_DEBUG_ASSERT((waitSize > 0), "waitSize is %u, it must be larger than 0", waitSize);
        ASCENDC_DEBUG_ASSERT(
            (waitSize <= GetBlockNum()), "waitSize is %u, max waitSize is %lld \n", waitSize, GetBlockNum());
        ASCENDC_DEBUG_ASSERT(
            (arriveSize <= GetBlockNum()), "waitSize is %u, max waitSize is %lld \n", arriveSize, GetBlockNum());
        return (arriveSize > waitSize) ? arriveSize * CACHE_LINE_LEN : waitSize * CACHE_LINE_LEN;
    }
}

template <PipeMode pipeMode>
__aicore__ inline void GroupBarrier<pipeMode>::__WriteCurrentValue(__gm__ BarrierInfo *BarrierInfoAddr)
{
    if ASCEND_IS_AIV {
        uint32_t num = (arriveSize >= waitSize) ? arriveSize : waitSize;
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventID);
        WaitFlag<HardEvent::S_MTE3>(eventID);
        SetAtomicAddImpl<int32_t>();
#if ASCENDC_CPU_DEBUG
        ProcessLock::GetProcessLock()->Write();
        __ubuf__ int32_t *dst =
            (__ubuf__ int32_t *)(GetTPipePtr()->GetBaseAddr((uint8_t)(TPosition::VECIN)) + UB_START_ADDR);
        copy_ubuf_to_gm((__gm__ void *)BarrierInfoAddr, (__ubuf__ void *)dst, 0, num, 1, 0, CACHELINE_BLKNUM - 1);
        ProcessLock::GetProcessLock()->Unlock();
#else
        __ubuf__ int32_t *dst = (__ubuf__ int32_t *)(UB_START_ADDR);
        // total: num * 32B block     src: consecutive   dst: apart by 512B
        copy_ubuf_to_gm((__gm__ void *)BarrierInfoAddr, (__ubuf__ void *)dst, 0, num, 1, 0, CACHELINE_BLKNUM - 1);
#endif
        SetAtomicNoneImpl();
    }
}
}  // namespace AscendC
#endif