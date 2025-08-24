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
 * \file kernel_operator_sync_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_SYNC_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SYNC_IMPL_H
namespace AscendC {
__aicore__ inline void ClcSyncCount(__gm__ int32_t* localSyncGM, __ubuf__ int32_t* ubWorkspaceAddr,
    const int32_t blockIdx, const int32_t totalBlocks, bool isFirst, int32_t& count)
{
    if (isFirst) {
        __ubuf__ int32_t* localUbAddr = ubWorkspaceAddr + (blockIdx * DEFAULT_BLK_NUM);
        *(reinterpret_cast<__ubuf__ int32_t*>(localUbAddr)) = 1;
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        copy_ubuf_to_gm(static_cast<__gm__ void*>(localSyncGM), static_cast<__ubuf__ void*>(localUbAddr), 0, 1, 1, 0,
            0);
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        count = 1;
        for (int32_t i = 0; i < totalBlocks; i++) {
            if (i != blockIdx) {
                count += *(reinterpret_cast<__ubuf__ int32_t*>(ubWorkspaceAddr) + i * ONE_BLK_FLOAT_NUM);
            }
        }
    } else {
        for (int32_t i = 0; i < totalBlocks; i++) {
            count += *(reinterpret_cast<__ubuf__ int32_t*>(ubWorkspaceAddr) + i * ONE_BLK_FLOAT_NUM);
        }
    }
}

__aicore__ inline int64_t GetBlockNum();
template <bool isAIVOnly = true>
__aicore__ inline void SoftSyncAllImpl(__gm__ int32_t* gmWorkspaceAddr, __ubuf__ int32_t* ubWorkspaceAddr,
    const int usedCores)
{
    if ASCEND_IS_AIC {
        return;
    }
    __sync_all_stub(usedCores, isAIVOnly);
    PipeBarrier<PIPE_ALL>();

    int32_t totalBlocks = isAIVOnly ? GetBlockNum() : (GetTaskRationImpl() * GetBlockNum());
    totalBlocks = usedCores != 0 ? usedCores : totalBlocks;
    int32_t blockIdx = isAIVOnly ? get_block_idx() : GetBlockIdxImpl();

    __gm__ int32_t* localSyncGM = gmWorkspaceAddr + (blockIdx * DEFAULT_BLK_NUM);
    __ubuf__ int32_t* localUbAddr = ubWorkspaceAddr + (blockIdx * DEFAULT_BLK_NUM);

    copy_gm_to_ubuf(static_cast<__ubuf__ void*>(localUbAddr), static_cast<__gm__ void*>(localSyncGM), 0, 1, 1, 0, 0);
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    int32_t curValue = *(reinterpret_cast<__ubuf__ int32_t*>(localUbAddr)) + 1;
    bool isFirst = curValue == 1 ? true : false;
    *(reinterpret_cast<__ubuf__ int32_t*>(localUbAddr)) = curValue;
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    copy_ubuf_to_gm(static_cast<__gm__ void*>(localSyncGM), static_cast<__ubuf__ void*>(localUbAddr), 0, 1, 1, 0, 0);
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    int32_t totalBlockCount = ONE_BLK_FLOAT_NUM * totalBlocks;
    uint16_t blockLen = totalBlockCount / AscendCUtils::GetC0Count(sizeof(int32_t));
    while (true) {
        copy_gm_to_ubuf(static_cast<__ubuf__ void*>(ubWorkspaceAddr), static_cast<__gm__ void*>(gmWorkspaceAddr), 0, 1,
            blockLen, 0, 0);
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        int32_t count = 0;
        ClcSyncCount(localSyncGM, ubWorkspaceAddr, blockIdx, totalBlocks, isFirst, count);
        event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        if (count >= (totalBlocks * curValue)) {
            break;
        }
    }
    __sync_all_stub(usedCores, isAIVOnly);
}

constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

__aicore__ inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

template<pipe_t AIV_PIPE = PIPE_MTE3, pipe_t AIC_PIPE = PIPE_FIX>
__aicore__ inline void SetNextTaskStartImpl()
{
    if ASCEND_IS_AIC {
        ffts_cross_core_sync(AIC_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG));
    }
    if ASCEND_IS_AIV {
        ffts_cross_core_sync(AIV_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
    }
    return;
}

__aicore__ inline void WaitPreTaskEndImpl()
{
    if ASCEND_IS_AIC {
        wait_flag_dev(AscendC::SYNC_AIC_FLAG);
        ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));
        wait_flag_dev(AscendC::SYNC_AIV_FLAG);
    }
    if ASCEND_IS_AIV {
        wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
        ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));
        wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
    }
    return;
}

template <bool isAIVOnly = true> __aicore__ inline void SyncAllImpl()
{
    PipeBarrier<PIPE_ALL>();
    if constexpr (isAIVOnly) {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
        return;
    }
    // trans software sync to hardware sync
    if ASCEND_IS_AIC {
        wait_flag_dev(SYNC_AIV_FLAG);
        ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
        wait_flag_dev(SYNC_AIC_FLAG);
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
        return;
    }
    if ASCEND_IS_AIV {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
        wait_flag_dev(SYNC_AIC_AIV_FLAG);
        return;
    }
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void NotifyEventImpl(uint16_t flagId)
{
    ffts_cross_core_sync(pipe, GetffstMsg(modeId, flagId));
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void WaitEventImpl(uint16_t flagId)
{
    (void)modeId;
    wait_flag_dev(flagId);
}

__aicore__ inline void SetSyncBaseAddrImpl(uint64_t config)
{
    set_ffts_base_addr(config);
}

__aicore__ inline void SetSyncBaseAddr(uint64_t config)
{
    SetSyncBaseAddrImpl(config);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SYNC_IMPL_H
