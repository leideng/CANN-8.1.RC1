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
 * \file kernel_operator_determine_compute_sync_impl.h
 * \brief
 */
#include "kernel_operator_common_intf.h"
#include "kernel_operator_vec_duplicate_intf.h"
#ifndef ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_IMPL_H
#define ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_IMPL_H

namespace AscendC {
__aicore__ inline void InitDetermineComputeWorkspaceCalc(GlobalTensor<int32_t> &gmWorkspace,
    LocalTensor<int32_t> &ubWorkspace)
{
    if ASCEND_IS_AIV {
        PipeBarrier<PIPE_ALL>();
        event_t eventID;
        auto blockNum = GetBlockNum();
        auto blockIdx = GetBlockIdx();
        if (GetBlockIdx() == 0) {
            Duplicate(ubWorkspace, 0, B32_DATA_NUM_PER_BLOCK * blockNum);
            eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventID);
            WaitFlag<HardEvent::V_MTE3>(eventID);
            DataCopy(gmWorkspace, ubWorkspace, B32_DATA_NUM_PER_BLOCK * blockNum);
        }
        ubWorkspace.SetValue(blockNum * B32_DATA_NUM_PER_BLOCK, 1);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline bool CheckUBWorkspace(LocalTensor<int32_t> &ubWorkspace, int64_t blockIdx, int64_t blockNum)
{
    int32_t repeatTime = ubWorkspace.GetValue(blockNum * B32_DATA_NUM_PER_BLOCK);
    int64_t offset = 0;
    // example: core num is n, current core id is i, current repeat time is k:
    // matched if workspace values are kkkk...k0...000: [k] * (i), [0] * (n-i)
    for (; offset < blockIdx * B32_DATA_NUM_PER_BLOCK; offset += B32_DATA_NUM_PER_BLOCK) {
        if (ubWorkspace.GetValue(offset) != repeatTime) {
            return false;
        }
    }
    for (; offset < blockNum * B32_DATA_NUM_PER_BLOCK; offset += B32_DATA_NUM_PER_BLOCK) {
        if (ubWorkspace.GetValue(offset) != 0) {
            return false;
        }
    }
    return true;
}

__aicore__ inline void WaitPreBlockCalc(GlobalTensor<int32_t> &gmWorkspace, LocalTensor<int32_t> &ubWorkspace)
{
    if ASCEND_IS_AIV {
        PipeBarrier<PIPE_ALL>();
        event_t eventID;
        auto blockIdx = GetBlockIdx();
        auto blockNum = GetBlockNum();
        bool matchFlag;
        do {
            DataCopy(ubWorkspace, gmWorkspace, blockNum * B32_DATA_NUM_PER_BLOCK);
            eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventID);
            WaitFlag<HardEvent::MTE2_S>(eventID);
            matchFlag = CheckUBWorkspace(ubWorkspace, blockIdx, blockNum);
            eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
            SetFlag<HardEvent::S_MTE2>(eventID);
            WaitFlag<HardEvent::S_MTE2>(eventID);
        } while (!matchFlag);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void NotifyNextBlockCalc(GlobalTensor<int32_t> &gmWorkspace, LocalTensor<int32_t> &ubWorkspace)
{
    if ASCEND_IS_AIV {
        PipeBarrier<PIPE_ALL>();
        event_t eventID;
        auto blockIdx = GetBlockIdx();
        auto blockNum = GetBlockNum();
        int32_t repeatTime = ubWorkspace.GetValue(blockNum * B32_DATA_NUM_PER_BLOCK);
        if (blockIdx + 1 == blockNum) {
            Duplicate(ubWorkspace, 0, blockNum * B32_DATA_NUM_PER_BLOCK);
            eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventID);
            WaitFlag<HardEvent::V_MTE3>(eventID);
            DataCopy(gmWorkspace, ubWorkspace, blockNum * B32_DATA_NUM_PER_BLOCK);
        } else {
            auto offset = blockIdx * B32_DATA_NUM_PER_BLOCK;
            eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventID);
            WaitFlag<HardEvent::S_V>(eventID);
            Duplicate(ubWorkspace[offset], repeatTime, B32_DATA_NUM_PER_BLOCK);
            eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventID);
            WaitFlag<HardEvent::V_MTE3>(eventID);
            DataCopy(gmWorkspace[offset], ubWorkspace[offset], B32_DATA_NUM_PER_BLOCK);
        }
        ubWorkspace.SetValue(blockNum * B32_DATA_NUM_PER_BLOCK, repeatTime + 1);
        PipeBarrier<PIPE_ALL>();
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_IMPL_H
