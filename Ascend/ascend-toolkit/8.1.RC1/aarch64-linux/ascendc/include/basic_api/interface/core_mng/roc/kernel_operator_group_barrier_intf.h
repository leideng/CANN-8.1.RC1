/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file kernel_operator_group_barrier_intf.h
 * \brief
 */

#ifndef ASCENDC_MODULE_OPERATOR_GROUP_BARRIER_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_GROUP_BARRIER_INTERFACE_H
#include "kernel_tensor.h"
#include "dav_c220/core_mng/roc/kernel_operator_cube_group_info.h"
namespace AscendC {
template <PipeMode pipeMode>
class GroupBarrier {
public:
    __aicore__ inline GroupBarrier(GM_ADDR groupWorkspace, uint32_t arriveSizeIn, uint32_t waitSizeIn);
    __aicore__ inline void Arrive(uint32_t arriveIndex);
    // stuck in while loop until all aiv has arrived, then update wait counter
    __aicore__ inline void Wait(uint32_t waitIndex);
    __aicore__ inline uint64_t GetWorkspaceLen();

private:
    __aicore__ inline void __WriteCurrentValue(__gm__ BarrierInfo *BarrierInfoAddr);
    __aicore__ inline GroupBarrier() = delete;
    __gm__ BarrierInfo *barrierInfoArrive;  // 64B in GM for storing current arrive counter
    __gm__ BarrierInfo *barrierInfoWait;    // 64B in GM for storing current wait counter
    uint32_t arriveSize;
    uint32_t waitSize;
    uint32_t counter;  // in which round
    bool hasArrive;    // whether current aiv has called arrive function in this round
};
}  // namespace AscendC
#endif