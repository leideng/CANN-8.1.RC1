/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file rac_server.h
 * \brief
 */
#ifndef RAC_SERVET_H
#define RAC_SERVET_H

#include "kernel_operator.h"
#include "../common.h"

namespace AscendC {
constexpr uint32_t AC_MSG_VALID_MASK = 0x5CDF123A;
enum AicpuComType {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_GATHER,
    HCCL_CMD_MAX
};

struct AivAicpuOpParam {
    AicpuComType commType; // 32b
    int32_t opType;   // 32b
    uint64_t sendBuffer;   //
    uint64_t recvBuffer;
    uint64_t count;
    uint64_t strideLen;

    // offset 32B
    int32_t hcclDataType;

    uint32_t valid;  // 检查消息有效性
    uint8_t isLast;  // 是否最后一个下
    uint8_t funID;   // 功能ID，1地址消息；  2开始工作
    uint8_t sendCnt; // 发送计数
    uint8_t rcvCnt;  // 执行结束轮次技术
    uint8_t everyTurnRsp; // 每轮都需要等待执行结束发送响应，再执行下一轮
    uint8_t everyTurnWait; // 每轮都需要等待work消息再执行
    uint8_t totalTurnCnt; // 总轮次
    uint8_t res[9];     // 整体消息64字节
};

#ifdef __CCE_AICORE__
#if __CCE_AICORE__ == 220
// Rich Communications Services
class HcclServer {
   public:
    __gm__ AivAicpuOpParam *msgSndWorkArea;
    __gm__ AivAicpuOpParam *msgRcvRspArea;
    uint8_t debugMode_;

    __aicore__ inline void Init(GM_ADDR win, uint8_t debugMode) {
        msgSndWorkArea = reinterpret_cast<__gm__ AivAicpuOpParam *>(win);
        msgRcvRspArea = reinterpret_cast<__gm__ AivAicpuOpParam *>(win + sizeof(AivAicpuOpParam));
        debugMode_ = debugMode;
    }

    __aicore__ inline void TurnNotifyRunSetMsg(uint32_t curTurn) {
        if (debugMode_ == static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_CUBE)) {
            return;
        }
        msgSndWorkArea->sendCnt = curTurn;
        msgSndWorkArea->valid = AC_MSG_VALID_MASK;
        AscendC::Barrier();
        dcci(reinterpret_cast<__gm__ int64_t *>(msgSndWorkArea), cache_line_t::SINGLE_CACHE_LINE,
             dcci_dst_t::CACHELINE_OUT);
    }

    __aicore__ inline void TurnWait(uint32_t totalTurn) {
        if (debugMode_ == static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_CUBE)) {
            return;
        }
        while (true) {
            AscendC::Barrier();
            dcci(reinterpret_cast<__gm__ int64_t *>(msgRcvRspArea), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
            if (msgRcvRspArea->rcvCnt >= totalTurn) {
                break;
            }
        }
        // Clear rcvCnt until all message received
        msgRcvRspArea->rcvCnt = 0;
        msgRcvRspArea->valid = ~AC_MSG_VALID_MASK;
        AscendC::Barrier();
        dcci(reinterpret_cast<__gm__ int64_t *>(msgRcvRspArea), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    }
};
#endif // __CCE_AICORE__ == 220

#if __CCE_AICORE__ < 220 // we can test this code in 220
constexpr uint32_t AC_MAX_AIV = 64;     // 最多有64个AIV
constexpr uint32_t HCCL_PARAM_SIZE = sizeof(AivAicpuOpParam); // must aligned by 32 bytes
// Rich Communications Services
class HcclServer {
public:
    __aicore__ inline HcclServer() = default;
    __ubuf__ AivAicpuOpParam *msgSndWorkArea;
    __ubuf__ AivAicpuOpParam *msgRcvRspArea;
    GlobalTensor<uint8_t> msgSndGlobalTensor;
    LocalTensor<uint8_t> msgSndLocalTensor;
    GlobalTensor<uint8_t> msgRcvGlobalTensor;
    LocalTensor<uint8_t> msgRcvLocalTensor;
    GlobalTensor<int32_t> syncGlobalTensor;
    LocalTensor<int32_t> syncLocalTensor;
    // 1:单独执行CUBE
    // 2:单独执行Vector
    // 4:单独执行AICPU KFC算子
    // 8:KFC等待通信结束
    // 16:KFC统计各阶段耗时
    uint8_t debugMode_;

    __aicore__ inline void ReadMsgFromGlobal(LocalTensor<uint8_t> local, GlobalTensor<uint8_t> global, uint32_t sizeInBytes) {
        DataCopy(local, global, sizeInBytes);
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
        set_flag(PIPE_MTE2, PIPE_S, eventID);
        wait_flag(PIPE_MTE2, PIPE_S, eventID);
    }

    __aicore__ inline void WriteMsgToGlobal(GlobalTensor<uint8_t> global, LocalTensor<uint8_t> local, uint32_t sizeInBytes) {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
        set_flag(PIPE_S, PIPE_MTE3, eventID);
        wait_flag(PIPE_S, PIPE_MTE3, eventID);
        DataCopy(global, local, sizeInBytes);
    }

    __aicore__ inline void Init(GM_ADDR win, uint8_t debugMode, TBuf<TPosition::VECCALC> &tmpBuf) {
        debugMode_ = debugMode;
        uint32_t offset = 0;
        msgSndGlobalTensor.SetGlobalBuffer(win + offset, HCCL_PARAM_SIZE);
        offset += HCCL_PARAM_SIZE;
        msgRcvGlobalTensor.SetGlobalBuffer(win + offset, HCCL_PARAM_SIZE);
        // use ubuf hold the result
        msgSndLocalTensor = tmpBuf.Get<uint8_t>();
        msgRcvLocalTensor = tmpBuf.Get<uint8_t>();
        msgSndWorkArea = reinterpret_cast<__ubuf__ AivAicpuOpParam *>(msgSndLocalTensor.GetPhyAddr());
        msgRcvRspArea = reinterpret_cast<__ubuf__ AivAicpuOpParam *>(msgRcvLocalTensor.GetPhyAddr());
    }

    __aicore__ inline void InitSoftSync(GM_ADDR softSyncOffset, uint32_t usedCoreNum,
            TBuf<TPosition::VECCALC> &tmpBuf) {
        const int32_t ELEMENT_CNT = DEFAULT_C0_SIZE / sizeof(int32_t);
        syncGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(softSyncOffset), usedCoreNum * ELEMENT_CNT);
        syncLocalTensor = tmpBuf.Get<int32_t>();
        // clear soft sync global tensor
        Duplicate(syncLocalTensor, 0, usedCoreNum * ELEMENT_CNT);
        pipe_barrier(PIPE_ALL);
        DataCopy(syncGlobalTensor, syncLocalTensor, usedCoreNum * ELEMENT_CNT);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void TurnNotifyRun(uint32_t blockIdx, uint32_t usedCoreNum, uint32_t curTurn) {
        if (debugMode_ == 1) {
            return;
        }
        // barrier to avoid ub conflict
        if (usedCoreNum > 1) {
            SyncAll(syncGlobalTensor, syncLocalTensor, usedCoreNum);
        }
        if (blockIdx == 0) {
            // barrier to avoid ub conflict
            pipe_barrier(PIPE_ALL);
            ReadMsgFromGlobal(msgSndLocalTensor, msgSndGlobalTensor, HCCL_PARAM_SIZE);
            msgSndWorkArea->sendCnt = curTurn;
            msgSndWorkArea->valid = AC_MSG_VALID_MASK;
            WriteMsgToGlobal(msgSndGlobalTensor, msgSndLocalTensor, HCCL_PARAM_SIZE);
        }
    }

    __aicore__ inline bool TurnWait(uint32_t totalTurn) {
        if (debugMode_ == 1) {
            return false;
        }
        pipe_barrier(PIPE_ALL);
        while (true) {
            ReadMsgFromGlobal(msgRcvLocalTensor, msgRcvGlobalTensor, HCCL_PARAM_SIZE);
            if (msgRcvRspArea->rcvCnt >= totalTurn) {
                break;
            }
        }
        // reset msgRcvRspArea
        msgRcvRspArea->rcvCnt = 0;
        msgRcvRspArea->valid = 0;
        WriteMsgToGlobal(msgRcvGlobalTensor, msgRcvLocalTensor, HCCL_PARAM_SIZE);
        return true;
    }
};
#endif //__CCE_AICORE__ < 220
#endif // __CCE_AICORE__

}
#endif // RAC_SERVET_H
