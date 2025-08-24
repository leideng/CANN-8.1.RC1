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
 * \file inner_kernel_operator_common_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_COMMON_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_COMMON_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_mm.h"

/*
 * ingroup：SetAtomicAdd
 * brief：Set the next data from UB to the outside of AI Core whether the move write Tensor operation performs
 * atomic accumulation.
 */
#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_set_atomic_impl.h"
#include "dav_c100/kernel_operator_common_impl.h"
#include "dav_c100/kernel_operator_vec_duplicate_impl.h"
#include "dav_c100/kernel_operator_sync_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_set_atomic_impl.h"
#include "dav_m200/kernel_operator_common_impl.h"
#include "dav_m200/kernel_operator_vec_duplicate_impl.h"
#include "dav_m200/kernel_operator_sync_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_set_atomic_impl.h"
#include "dav_c220/kernel_operator_common_impl.h"
#include "dav_c220/kernel_operator_sync_impl.h"
#include "dav_c220/kernel_operator_vec_duplicate_impl.h"
#include "dav_c220/kfc/kfc_comm_client.h"
#include "dav_c220/kfc/kfc_comm_server.h"
#include "dav_c220/core_mng/roc/kernel_operator_cube_group_handle_impl.h"
#include "dav_c220/core_mng/roc/kernel_operator_group_barrier_impl.h"
#endif
#include "impl/kernel_pop_stack_buffer.h"

namespace AscendC {
/*
 * @ingroup：IBSet, IBWait
 * @brief：Set the flag bit of a core
 * @param [in] gmWorkspace GlobalTensor to store core state
 * @param [in] ubWorkspce LocalTensor for current core
 * @param [in] blockIdx the idx number waiting for the core
 * @param [in] eventID Set and wait events
 */
__aicore__ inline int64_t GetBlockNum();
template <bool isAIVOnly>
__aicore__ inline void IBSet(const GlobalTensor<int32_t> &gmWorkspace,
    const LocalTensor<int32_t> &ubWorkspace, int32_t blockIdx, int32_t eventID)
{
    int32_t blockNum = GetBlockNum();
#if __CCE_AICORE__ == 220
    if ASCEND_IS_AIC {
        return;
    }
    if (!isAIVOnly) {
        blockNum = GetBlockNum() * 2;
    }
#endif
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 200
    __ib_set_stub(blockIdx, eventID, isAIVOnly);
#endif
    auto localSyncGM = gmWorkspace[blockNum * 8 * eventID + blockIdx * 8];
    pipe_barrier(PIPE_ALL);

    while (true) {
        DataCopy(ubWorkspace, localSyncGM, ONE_BLK_SIZE / sizeof(int32_t));
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        if (ubWorkspace.GetValue(0) == 0) {
            ubWorkspace.SetValue(0, 1);
            event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            DataCopy(localSyncGM, ubWorkspace, ONE_BLK_SIZE / sizeof(int32_t));
            break;
        }
    }
    pipe_barrier(PIPE_ALL);
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 200
    __ib_set_stub(blockIdx, eventID, isAIVOnly);
#endif
}

template <bool isAIVOnly>
__aicore__ inline void IBWait(const GlobalTensor<int32_t> &gmWorkspace,
    const LocalTensor<int32_t> &ubWorkspace, int32_t blockIdx, int32_t eventID)
{
    int32_t blockNum = GetBlockNum();
#if __CCE_AICORE__ == 220
    if ASCEND_IS_AIC {
        return;
    }
    if (!isAIVOnly) {
        blockNum = GetBlockNum() * 2;
    }
#endif
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 200
    __ib_wait_stub(blockIdx, eventID, isAIVOnly);
#endif
    auto localSyncGM = gmWorkspace[blockNum * 8 * eventID + blockIdx * 8];
    pipe_barrier(PIPE_ALL);

    while (true) {
        DataCopy(ubWorkspace, localSyncGM, ONE_BLK_SIZE / sizeof(int32_t));
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        if (ubWorkspace.GetValue(0) == 1) {
            ubWorkspace.SetValue(0, 0);
            event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            DataCopy(localSyncGM, ubWorkspace, ONE_BLK_SIZE / sizeof(int32_t));
            break;
        }
    }
    pipe_barrier(PIPE_ALL);
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 200
    __ib_wait_stub(blockIdx, eventID, isAIVOnly);
#endif
}

/*
 * @ingroup：SyncALL
 * @brief：Set flag bits of all cores
 * @param [in] gmWorkspace GlobalTensor to store core state
 * @param [in] ubWorkspce LocalTensor for current core
 */
template <bool isAIVOnly>
__aicore__ inline void SyncAll(const GlobalTensor<int32_t> &gmWorkspace,
    const LocalTensor<int32_t> &ubWorkspace, const int usedCores)
{
#if ASCENDC_CPU_DEBUG
    SoftSyncAllImpl<false>((__gm__ int32_t*)gmWorkspace.GetPhyAddr(),
        (__ubuf__ int32_t*)ubWorkspace.GetPhyAddr(), usedCores);
#else
    SoftSyncAllImpl<isAIVOnly>((__gm__ int32_t*)gmWorkspace.GetPhyAddr(),
        (__ubuf__ int32_t*)ubWorkspace.GetPhyAddr(), usedCores);
#endif
}

__aicore__ inline int64_t GetBlockIdx()
{
    return GetBlockIdxImpl();
}

__aicore__ inline int64_t GetBlockNum()
{
    return get_block_num();
}

__aicore__ inline int64_t GetSubBlockIdx()
{
    return GetSubBlockIdxImpl();
}

__aicore__ inline int64_t GetTaskRation()
{
    return GetTaskRationImpl();
}

template <typename T>
__aicore__ inline __in_pipe__(V)
    __out_pipe__(MTE3) void InitOutput(GlobalTensor<T> gmWorkspaceAddr, uint32_t size, T value)
{
#if __CCE_AICORE__ == 220
    if ASCEND_IS_AIC {
        return;
    }
    LocalTensor<T> popBuffer;
    bool ret = PopStackBuffer<T, TPosition::LCM>(popBuffer);
    uint32_t maxBurstSize = (MAX_REPEAT_TIMES * ONE_BLK_SIZE) / sizeof(T);
    uint32_t popSize = popBuffer.GetSize() >= maxBurstSize ? maxBurstSize : popBuffer.GetSize();
    uint32_t round = size / popSize;
    uint32_t tail = size % popSize;
    uint32_t roundSize = round != 0 ? popSize : 0;
    DuplicateImpl<T>((__ubuf__ T*)popBuffer.GetPhyAddr(), value, popSize);
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    struct DataCopyExtParams repeatParams;
    uint32_t comOffset = 0;
    // compute the main block
    repeatParams = { 1, static_cast<uint32_t>(roundSize * sizeof(T)), 0, 0, 0 };
    for (int index = 0; index < round; ++index) {
        DataCopyPadUB2GMImpl((__gm__ T*)gmWorkspaceAddr.GetPhyAddr() + comOffset,
            (__ubuf__ T*)popBuffer.GetPhyAddr(),
            repeatParams);
        comOffset += roundSize;
    }
    // compute the tail block
    repeatParams = {1, static_cast<uint32_t>(tail * sizeof(T)), 0, 0, 0};
    if (tail != 0) {
        comOffset = round * roundSize;
        DataCopyPadUB2GMImpl((__gm__ T*)gmWorkspaceAddr.GetPhyAddr() + comOffset,
            (__ubuf__ T*)popBuffer.GetPhyAddr(),
            repeatParams);
    }
#endif
}

template<bool isAIVOnly>
__aicore__ inline void SyncAll()
{
    SyncAllImpl<isAIVOnly>();
}

template <AtomicDtype type, AtomicOp op>
__aicore__ inline void SetStoreAtomicConfig()
{
    SetStoreAtomicConfigImpl<static_cast<atomic_type_t>(type), static_cast<atomic_op_t>(op)>();
}

__aicore__ inline int64_t GetStoreAtomicConfig()
{
    return GetStoreAtomicConfigImpl();
}

__aicore__ inline void GetStoreAtomicConfig(uint16_t &atomicType, uint16_t &atomicOp)
{
    GetStoreAtomicConfigImpl(atomicType, atomicOp);
}

template <pipe_t pipe>
__aicore__ inline void NotifyEvent(uint16_t flagId)
{
    constexpr uint8_t subBlockSyncMode = 0x02;
    NotifyEventImpl<subBlockSyncMode, pipe>(flagId);
}

template <pipe_t pipe=PIPE_S>
__aicore__ inline void WaitEvent(uint16_t flagId)
{
    constexpr uint8_t mode = 0;
    WaitEventImpl<mode, pipe>(flagId);
}

template<uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
{
    NotifyEventImpl<modeId, pipe>(flagId);    
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
{
    WaitEventImpl<modeId, pipe>(flagId);
}

template <typename T>
__aicore__ inline void DataCachePreload(const GlobalTensor<uint64_t> &srcTensor, const T cacheOffset)
{
    DataCachePreloadImpl(srcTensor, cacheOffset);
}

__aicore__ inline void ICachePreLoad(const int64_t preFetchLen)
{
    PreLoad(preFetchLen);
}

__aicore__ inline int64_t GetICachePreloadStatus()
{
    return GetICachePreloadStatusImpl();
}

__aicore__ inline void CheckLocalMemoryIA(const CheckLocalMemoryIAParam& checkParams)
{
    CheckLocalMemoryIAImpl(checkParams);
}

#if __CCE_AICORE__ >= 220
template <HardEvent event, MemoryT memT, bool isVirtual> __aicore__ inline void HSetFlag(int32_t eventID)
{
    if (g_coreType == AIV) {
        return;
    }
    HSetFlagImpl<event, memT, isVirtual>(eventID);
}

template <HardEvent event, MemoryT memT, bool isVirtual> __aicore__ inline void HWaitFlag(int32_t eventID)
{
    if (g_coreType == AIV) {
        return;
    }
    HWaitFlagImpl<event, memT, isVirtual>(eventID);
}
#endif

} // namespace AscendC
#endif // ASCENDC_MODULE_INNER_OPERATOR_COMMON_INTERFACE_H
