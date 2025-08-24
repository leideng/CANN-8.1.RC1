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
 * \file kernel_operator_common_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
#define ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
#include "kernel_utils.h"
#include "kernel_struct_mm.h"
namespace AscendC {
__aicore__ inline int64_t GetSubBlockIdxImpl()
{
    return 0;
}

__aicore__ inline int64_t GetTaskRationImpl()
{
    return 1;
}

__aicore__ inline int64_t GetBlockIdxImpl()
{
    return get_block_idx();
}

[[deprecated(
    "NOTICE: SetSysWorkSpace has been deprecated and will be removed in the next version.")]]
__aicore__ inline void SetSysWorkspace(GM_ADDR workspace)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASSERT((workspace != nullptr) && "workspace can not be nullptr");
#else
    if (g_sysWorkspaceReserved == nullptr) {
        g_sysWorkspaceReserved = workspace;
    }
#endif
}

__aicore__ inline void SetSysWorkspaceForce(GM_ADDR workspace)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASSERT((workspace != nullptr) && "workspace can not be nullptr");
#else
    g_sysWorkspaceReserved = workspace;
#endif
}

__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASSERT((workspace != nullptr) && "workspace can not be nullptr");
    return workspace;
#else
    (void)(workspace);
    // reserved 16 * 1024 * 1024 Bytes
    return g_sysWorkspaceReserved + RESERVED_WORKSPACE;
#endif
}

template<pipe_t AIV_PIPE = PIPE_MTE3, pipe_t AIC_PIPE = PIPE_FIX>
__aicore__ inline void SetNextTaskStartImpl()
{
    ASSERT(0 && "unsupported SetNextTaskStart on current device");
}

__aicore__ inline void WaitPreTaskEndImpl()
{
    ASSERT(0 && "unsupported WaitPreTaskEnd on current device");
}

template<bool isAIVOnly = true>
__aicore__ inline void SoftSyncAllImpl(__gm__ int32_t* gmWorkspaceAddr, __ubuf__ int32_t* ubWorkspaceAddr,
    const int usedCores)
{
    ASSERT(0 && "unsupported software syncAll on current device");
}

template<bool isAIVOnly = true>
__aicore__ inline void SyncAllImpl()
{
    ASSERT(0 && "unsupported hardware syncAll on current device");
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void NotifyEventImpl(uint16_t flagId)
{
    ASCENDC_ASSERT((false), "CrossCoreSetFlag is not supported on current device");
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void WaitEventImpl(uint16_t flagId)
{
    ASCENDC_ASSERT((false), "CrossCoreWaitFlag is not supported on current device");
}

template <atomic_type_t type, atomic_op_t op>
__aicore__ inline void SetStoreAtomicConfigImpl()
{
    set_st_atomic_cfg(type, op);
}

__aicore__ inline int64_t GetStoreAtomicConfigImpl()
{
    return get_st_atomic_cfg();
}

__aicore__ inline void GetStoreAtomicConfigImpl(uint16_t &atomicType, uint16_t &atomicOp)
{
    int64_t stAtomic = get_st_atomic_cfg();
    constexpr uint64_t typeMask = 0x7;
    constexpr uint64_t opBit = 4;
    constexpr uint64_t opMask = 0x3;
    atomicType = (static_cast<uint64_t>(stAtomic) & typeMask);
    atomicOp = ((static_cast<uint64_t>(stAtomic) >> opBit) & opMask);
}

__aicore__ inline void SetSyncBaseAddr(uint64_t config)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "SetSyncBaseAddr is not supported on current device"); });
}

template <typename T>
__aicore__ inline void DataCachePreloadImpl(const GlobalTensor<uint64_t> &srcTensor, const T cacheOffset)
{
    if constexpr ((IsSameType<T, int16_t>::value) || (IsSameType<T, int64_t>::value)) {
        dc_preload((__gm__ uint64_t *)srcTensor.GetPhyAddr(), cacheOffset);
    } else {
        static_assert(SupportType<T, int16_t, int64_t>(),
                      "current cacheOffset data type is not supported on current device.");
    }
}

__aicore__ inline int64_t GetICachePreloadStatusImpl()
{
    static_assert((__CCE_AICORE__ == 300), "unsupport GetICachePreloadStatus on current device");
    return 0;
}

__aicore__ inline void PreLoad(const int64_t preFetchLen)
{
    ASCENDC_ASSERT((false), "ICachePreLoad is not supported on current device");
}

__aicore__ inline void CheckLocalMemoryIAImpl(const CheckLocalMemoryIAParam& checkParams)
{
    uint64_t config = 0;
    config = config | (static_cast<uint64_t>(checkParams.startAddr) << 48);     // start address, DEC[63:48]
    config = config | (static_cast<uint64_t>(checkParams.endAddr) << 32);       // end address, DEC[47:32]
    config = config | (static_cast<uint64_t>(checkParams.isScalarRead) << 31);  // scalar read access, DEC[31]
    config = config | (static_cast<uint64_t>(checkParams.isScalarWrite) << 30); // scalar write access, DEC[30]
    config = config | (static_cast<uint64_t>(checkParams.isVectorRead) << 29);  // vector read access, DEC[29]
    config = config | (static_cast<uint64_t>(checkParams.isVectorWrite) << 28); // vector write access, DEC[28]
    config = config | (static_cast<uint64_t>(checkParams.isMteRead) << 27);     // vector mte read access, DEC[27]
    config = config | (static_cast<uint64_t>(checkParams.isMteWrite) << 26);    // vector mte write access, DEC[26]
    config = config | (checkParams.reserved << 1);                              // reserved, DEC[25:1]
    config = config | (static_cast<uint8_t>(checkParams.isEnable));             // enable bit, DEC[0]
    if (checkParams.enableBit == SET_DATA_EXP_ZERO) {
        set_data_exp_0(config);
    } else if (checkParams.enableBit == SET_DATA_EXP_ONE) {
        set_data_exp_1(config);
    } else if (checkParams.enableBit == SET_DATA_EXP_TWO) {
        set_data_exp_2(config);
    } else if (checkParams.enableBit == SET_DATA_EXP_THREE) {
        set_data_exp_3(config);
    } else {
        static_assert((__CCE_AICORE__ == 300), "unsupport this enableBit on current device");
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
