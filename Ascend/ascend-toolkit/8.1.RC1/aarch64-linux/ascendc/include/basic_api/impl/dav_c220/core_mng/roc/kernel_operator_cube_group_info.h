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
 * \file kernel_operator_cube_group_info.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_CUBE_GROUP_INFO_H
#define ASCENDC_MODULE_OPERATOR_CUBE_GROUP_INFO_H
namespace AscendC {

constexpr uint32_t MAX_MSG_PER_AIV = 4;   // for 1 aic and 1 aiv, table stores at most 4 message
constexpr uint32_t BARRIER_SIZE = 64;     // 1 barrier for 1 64B due to dcci
constexpr uint32_t BARRIER_MAX_AIV = 50;  // at most 50 aiv
constexpr uint16_t CACHE_LINE_LEN = 512;  // cacheline length is 512B
constexpr uint32_t UB_START_ADDR = TOTAL_UB_SIZE - ONE_BLK_SIZE * BARRIER_MAX_AIV;  // GroupBarrier start address in UB
constexpr uint16_t CACHELINE_BLKNUM = CACHE_LINE_LEN / ONE_BLK_SIZE;                // 1 cacheline = n * 32B block

enum class CubeMsgState : uint8_t {
    FREE = 0,  // current CubeMsg is empty, allow to AllocMessage
    VALID,     // current CubeMsg needs aic to read and execute
    QUIT,      // tell aic that one aiv has ended service
    FAKE       // current CubeMsg is fake, need aic to FREE that msg when reading msg with skipCnt!=0
               // ex: aic read aiv0 skipCnt = 4, then aiv1~aiv4 need to be set FAKE first, then aic set to FREE.
};

struct CubeGroupMsgHead {                                 // 2B
    volatile CubeMsgState msgState = CubeMsgState::FREE;  // indicate aic / aiv current status
    volatile uint8_t aivID;
};

struct BarrierInfo {
    volatile uint32_t head;  // counter value for Arrive / Wait
    uint32_t buffer[15];     // gurantee 64B aligned
};

// method to update arrive / wait counter
enum class PipeMode : uint8_t { SCALAR_MODE = 0, MTE3_MODE = 1, MAX };

template <int32_t ActualFuncId, int32_t ExpectFuncId>
struct IsEqual {};

template <int32_t FuncId>
struct IsEqual<FuncId, FuncId> {
    using Type = void;
};
}  // namespace AscendC
#endif