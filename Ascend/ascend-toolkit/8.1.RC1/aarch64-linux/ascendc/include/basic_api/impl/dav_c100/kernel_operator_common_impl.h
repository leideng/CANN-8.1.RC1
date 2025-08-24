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
    return block_idx;
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

template <atomic_type_t type, atomic_op_t op>
__aicore__ inline void SetStoreAtomicConfigImpl()
{
    ASCENDC_ASSERT((false), "SetStoreAtomicConfig is not supported on current device");
}

__aicore__ inline int64_t GetStoreAtomicConfigImpl()
{
    static_assert((__CCE_AICORE__ == 100), "GetStoreAtomicConfig is not supported on current device");
    return 0;
}

__aicore__ inline void GetStoreAtomicConfigImpl(uint16_t &atomicType, uint16_t &atomicOp)
{
    ASCENDC_ASSERT((false), "GetStoreAtomicConfig is not supported on current device");
}

template <typename T>
__aicore__ inline void DataCachePreloadImpl(const GlobalTensor<uint64_t> &srcTensor, const T cacheOffset)
{
    static_assert((__CCE_AICORE__ == 100), "unsupport DataCachePreload on current device");
}

__aicore__ inline int64_t GetICachePreloadStatusImpl()
{
    static_assert((__CCE_AICORE__ == 100), "unsupport GetICachePreloadStatus on current device");
    return 0;
}

__aicore__ inline void PreLoad(const int64_t preFetchLen)
{
    ASCENDC_ASSERT((false), "ICachePreLoad is not supported on current device");
}

__aicore__ inline void CheckLocalMemoryIAImpl(const CheckLocalMemoryIAParam& checkParams)
{
    static_assert((__CCE_AICORE__ == 100), "unsupport this enableBit on current device");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
