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
 * \file kernel_operator_common_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_COMMON_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_COMMON_INTERFACE_H
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
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_set_atomic_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_set_atomic_impl.h"
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
template <bool isAIVOnly = true>
__aicore__ inline void IBSet(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace,
                                  int32_t blockIdx, int32_t eventID);

template <bool isAIVOnly = true>
__aicore__ inline void IBWait(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace,
                                   int32_t blockIdx, int32_t eventID);

/*
 * @ingroup：SyncALL
 * @brief：Set flag bits of all cores
 * @param [in] gmWorkspace GlobalTensor to store core state
 * @param [in] ubWorkspce LocalTensor for current core
 */
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace,
                                 const int32_t usedCores = 0);

__aicore__ inline int64_t GetBlockIdx();

__aicore__ inline int64_t GetBlockNum();

__aicore__ inline int64_t GetSubBlockIdx();

__aicore__ inline int64_t GetTaskRation();

template <typename T>
__aicore__ inline __in_pipe__(V)
    __out_pipe__(MTE3) void InitOutput(GlobalTensor<T> gmWorkspaceAddr, uint32_t size, T value = 0);

template <bool isAIVOnly = true>
__aicore__ inline void SyncAll();

enum class AtomicDtype { ATOMIC_NONE = 0, ATOMIC_F32, ATOMIC_F16, ATOMIC_S16, ATOMIC_S32, ATOMIC_S8, ATOMIC_BF16 };

enum class AtomicOp { ATOMIC_SUM = 0 };

template <AtomicDtype type, AtomicOp op>
__aicore__ inline void SetStoreAtomicConfig();

__aicore__ inline void GetStoreAtomicConfig(uint16_t& atomicType, uint16_t& atomicOp);

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId);

template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);

template <typename T>
__aicore__ inline void DataCachePreload(const GlobalTensor<uint64_t>& srcTensor, const T cacheOffset);

__aicore__ inline int64_t GetICachePreloadStatus();

__aicore__ inline void ICachePreLoad(const int64_t preFetchLen);

__aicore__ inline void CheckLocalMemoryIA(const CheckLocalMemoryIAParam& checkParams);
}  // namespace AscendC
#endif  // ASCENDC_MODULE_OPERATOR_COMMON_INTERFACE_H
