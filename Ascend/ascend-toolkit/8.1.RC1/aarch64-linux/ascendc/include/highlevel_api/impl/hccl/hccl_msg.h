/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_msg.h
 * \brief
 */
#ifndef IMPL_HCCL_HCCL_MSG_H
#define IMPL_HCCL_HCCL_MSG_H

namespace AscendC {
constexpr int32_t HCCL_FAILED = -1;
constexpr int32_t HCCL_SUCCESS = 0;
constexpr int32_t HCCL_MAX_HANDLE_ID = 63;
constexpr int8_t INVALID_HANDLE_ID = -1;
constexpr int8_t INVALID_MSG_POSITION = -1;

constexpr uint32_t HCCL_MAX_RANK_NUM = 32U;
constexpr uint32_t HCCL_MAX_RANK_NUM_V2 = 256;
constexpr uint32_t HCCL_MSG_CNT = 64;
constexpr uint32_t HCCL_MSG_VALID_MASK = 0x5CDF123A;

constexpr uint32_t HCCL_CCTILING_SIZE = 280;
constexpr uint32_t HCCL_CMD_TYPE_OFFSET = HCCL_CCTILING_SIZE - 8;
constexpr uint32_t HCCL_ALG_NAME_OFFSET = HCCL_CMD_TYPE_OFFSET - 128U;
constexpr uint32_t HCCL_STEP_SIZE_OFFSET = 2U;

constexpr uint32_t U64_CNT_PER_CACHELINE = 8U;
constexpr uint8_t HCCL_MSG_EXT_RESERVED_CNT = 6U;
constexpr uint32_t HCCL_VALID_POS = 12U;
constexpr uint32_t HCCL_MSG_DATA_CNT = 16U;
// Used to calc xor checksum for HcclMsg
struct DataBlock {
    uint32_t data[HCCL_MSG_DATA_CNT];
};

// 32 bytes aligned if using ubuf and dma to send/recv
// 64 bytes aligned if using scalar to write/read
struct V0MsgAdditionInfo {
    HcclDataType hcclDataType;
    uint32_t p2pSrcDestRankId;  // RankId of the peer end of send/recv, destRank for send, srcRank for recv
    uint32_t valid;             // msg valid when setting as HCCL_MSG_VALID_MASK
    uint8_t repeatCnt;          // The number of comm task launched by this msg is repeatCnt. The default is 1.
    uint8_t everyTurnRsp;       // Wait for the current turn to finish and a response before the next turn is executed
    uint8_t everyTurnWait;      // Each turn needs to wait for the work message before execution
    int8_t commDepGroupID;      // The comm group id that needs to wait for the execution of this msg. -1 default,
                                // indicating no need to wait.
    HcclHandle commDepHandleID; // The comm task of handleId needed to wait for the execution of this msg. -1 default,
                                // indicating no need to wait.
    HcclHandle selfHandleID;    // handleId of this comm msg, -1 for control msg.
    uint8_t seqNum;
    uint8_t version;
    uint32_t xorCheck;          // xor checksum
};

struct V1MsgAdditionInfo {
    uint64_t ccOpTilingData;
    uint32_t valid;             // msg valid when setting as HCCL_MSG_VALID_MASK
    HcclDataType hcclDataType;
    uint8_t repeatCnt;          // The number of comm task launched by this msg is repeatCnt. The default is 1.
    HcclHandle selfHandleID;    // handleId of this comm msg, -1 for control msg.
    uint8_t seqNum;
    uint8_t version;
    uint32_t xorCheck;          // xor checksum
};

struct HcclMsg {
    HcclCMDType commType;       // comm primitive type，AllReduce/AllGather.../Finalize/InterHcclGroupSync
    HcclReduceOp opType;        // reduce op type，sum/prod/max/min
    uint64_t sendBuffer;        // src buffer addr
    uint64_t recvBuffer;        // dst buffer addr
    uint64_t dataCnt;           // number of data participating in comm task
    uint64_t strideCount;       // Communication and computing fusion scenario will involve tiling,
                                // which may lead to data discontinuity.
                                // Thus, use strideCount filed to describe the offset of each data-block
                                // in discontinuous memory.
    union {
        V0MsgAdditionInfo v0Msg;
        V1MsgAdditionInfo v1Msg;
    } addMsg;
};

// HcclMsgExt is only used by AlltoAllV, and is separate from HcclMsg to improve read/write performance of HcclMsg.
// Current HcclMsgExt support 256 ranks max.
// Current size of HcclMsgExt is 8256B, while stack frame size is 32KB limited. Thus, do not define HcclMsgExt object.
struct HcclMsgExt {
    // sendCounts[i] represents the data count sent to rank i by this rank.
    uint64_t sendCounts[HCCL_MAX_RANK_NUM_V2];
    // sendOffset[i] represents the offset count of the data sent to rank i by this rank relative to sendBuf.
    uint64_t sendOffset[HCCL_MAX_RANK_NUM_V2];
    // recvCounts[i] represents the data count received from rank i to this rank.
    uint64_t recvCounts[HCCL_MAX_RANK_NUM_V2];
    // recvOffset[i] represents the offset count of the data received from rank i to this rank relative to recvBuf.
    uint64_t recvOffset[HCCL_MAX_RANK_NUM_V2];
    uint64_t reserved[HCCL_MSG_EXT_RESERVED_CNT];  // cacheline aligned for valid and xorCheck
    uint64_t valid;     // set by api, reset by server
    uint64_t xorCheck;  // set by api, checked by server to ensure msg integrity
};

struct AlltoAllVParamExt {
    uint64_t *sendCounts;
    uint64_t *sdispls;
    uint64_t *recvCounts;
    uint64_t *rdispls;
    __aicore__ inline void AssembleHcclMsgExt(uint32_t rankDim, __gm__ HcclMsgExt *dst) const {
        uint64_t xorCheck = 0U;
        for (uint32_t i = 0U; i < rankDim; ++i) {
            xorCheck ^= dst->sendCounts[i] = sendCounts[i];
            xorCheck ^= dst->sendOffset[i] = sdispls[i];
            xorCheck ^= dst->recvCounts[i] = recvCounts[i];
            xorCheck ^= dst->recvOffset[i] = rdispls[i];
        }
        dst->xorCheck = (xorCheck ^ HCCL_MSG_VALID_MASK);
        dst->valid = HCCL_MSG_VALID_MASK;
    }
};

constexpr uint64_t COMMIT_VALID_MASK = 987654321U;   // commit msg valid mask
constexpr uint64_t FINALIZE_FINISH_CNT = 1234567899999999999UL;  // server write finish msg when all hccl task finished

// cacheline size aligned by 64 bytes
struct TurnCnt {
    uint64_t valid;       // COMMIT_VALID_MASK, writen by client when Commit, checked by server
    uint64_t cnt;         // commit cnt, writen by client, reset by server
    uint64_t reserved[6];
};

struct ControlHcclMsg {
    uint8_t restart;
    uint8_t restarting;
    uint8_t restartCnt;
    uint8_t resetSeq;
    uint8_t reserved[60];
};

constexpr uint32_t BYTE_PER_KB = 1024U;
constexpr uint32_t BYTE_PER_MB = BYTE_PER_KB * BYTE_PER_KB;
// Current HcclMsgArea use count mode. Two msg bodies are used, one for read and one for write, to avoid aicore and
// aicpu reading or writing sendcnt/recvcnt at the same time.
// If using msg queue mode, then the state change can be in one msg, because it will not be written simultaneously.
// HcclMsgArea is the 16MB space reserved by workspace in struct HcclCombinOpParam and belongs to each comm group.
struct HcclMsgArea {
    HcclMsg sendMsgs[HCCL_MSG_CNT];
    HcclMsg recvMsgs[HCCL_MSG_CNT];
    uint8_t reserved0[8 * BYTE_PER_KB];    // for abi compatibility

    // commitTurnCnt and sendMsgList correspond one-to-one to inform the server times the task needs to be executed.
    // Ascend 910B and Ascend 310P support repeat>1 scenarios, so the element values are 1~repeat by Commit.
    // ccu does not support repeat>1 scenarios, so the element value can only be written to 1 by Commit.
    // The use of uint64_t is compatible with the requirement that ccu monitor 64bit.
    TurnCnt commitTurnCnt[HCCL_MSG_CNT];    // writen by client, indicating task num needed to be executed.
    TurnCnt finishedTurnCnt[HCCL_MSG_CNT];  // writen by server, indicating task num has been executed.
    uint8_t reserved1[BYTE_PER_MB];
    HcclMsgExt paramExtMsgList[HCCL_MSG_CNT];
    ControlHcclMsg controlMsg;
};

struct CommonPrepareParam {
    HcclCMDType commType;
    GM_ADDR sendBuf;
    GM_ADDR recvBuf;
    uint64_t count;
    HcclDataType dataType;
    HcclReduceOp op;
    uint64_t strideCount;
    uint8_t repeat = 1U;
    AlltoAllVParamExt paramExt; // only used by AlltoAllV

    __aicore__ inline void AssembleHcclMsg(
        int8_t ver, HcclHandle handle, uint64_t tiling, __gm__ HcclMsg *dst, __gm__ HcclMsgArea *hcclMsgArea_) const
    {
        HcclMsg tmp;
        static uint8_t primitiveId = 0U;
#if defined(AICORE_EXCEPTION_RESTART) && AICORE_EXCEPTION_RESTART == 1
        __gm__ ControlHcclMsg *controlMsgGM = &hcclMsgArea_->controlMsg;
        dcci(reinterpret_cast<__gm__ int64_t *>(controlMsgGM), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        if (controlMsgGM->resetSeq > 0) {
            controlMsgGM->resetSeq = 0;
            primitiveId = 0U;
        }
#endif
        tmp.commType = commType;
        if (commType == HcclCMDType::HCCL_CMD_FINALIZE) {
            primitiveId = 0U;
            if (ver != 0) {
                tmp.addMsg.v1Msg.ccOpTilingData = 0UL;
            }
        } else {
            tmp.opType = op;
            tmp.sendBuffer = reinterpret_cast<uint64_t>(sendBuf);
            tmp.recvBuffer = reinterpret_cast<uint64_t>(recvBuf);
            tmp.dataCnt = count;
            tmp.strideCount = strideCount;
            if (ver == 0) {
                tmp.addMsg.v0Msg.hcclDataType = dataType;
                tmp.addMsg.v0Msg.repeatCnt = repeat;
                tmp.addMsg.v0Msg.selfHandleID = handle;
                tmp.addMsg.v0Msg.seqNum = primitiveId++;
                tmp.addMsg.v0Msg.version = ver;
            } else {
                tmp.addMsg.v1Msg.ccOpTilingData = tiling;
                tmp.addMsg.v1Msg.hcclDataType = dataType;
                tmp.addMsg.v1Msg.repeatCnt = repeat;
                tmp.addMsg.v1Msg.selfHandleID = handle;
                tmp.addMsg.v1Msg.seqNum = primitiveId++;
                tmp.addMsg.v1Msg.version = ver;
            }
        }
        if (ver == 0) {
            tmp.addMsg.v0Msg.valid = HCCL_MSG_VALID_MASK;
        } else {
            tmp.addMsg.v1Msg.valid = HCCL_MSG_VALID_MASK;
        }
        CopyHcclMsg(reinterpret_cast<const uint8_t *>(&tmp), dst);
    }

    __aicore__ inline void AssembleHcclMsg(int8_t srcGroupID, HcclHandle srcHandleID, __gm__ HcclMsg *dst) const {
        HcclMsg tmp;
        tmp.commType = commType;
        tmp.addMsg.v0Msg.commDepGroupID = srcGroupID;
        tmp.addMsg.v0Msg.commDepHandleID = srcHandleID;
        tmp.addMsg.v0Msg.valid = HCCL_MSG_VALID_MASK;
        CopyHcclMsg(reinterpret_cast<const uint8_t *>(&tmp), dst);
    }

    __aicore__ inline void CopyHcclMsg(const uint8_t *src, __gm__ HcclMsg *dst) const {
        __gm__ DataBlock *tmpDst = reinterpret_cast<__gm__ DataBlock *>(dst);
        volatile uint32_t xorCheck = 0U;
        for (uint32_t i = 0; i < HCCL_MSG_DATA_CNT - 1U; ++i) {
            if (i == HCCL_VALID_POS) {
                xorCheck ^= HCCL_MSG_VALID_MASK;
            } else {
                xorCheck ^= tmpDst->data[i] = *(reinterpret_cast<const uint32_t *>(src));
            }
            src += sizeof(tmpDst->data[i]);
        }
        tmpDst->data[HCCL_MSG_DATA_CNT - 1U] = xorCheck;
        tmpDst->data[HCCL_VALID_POS] = HCCL_MSG_VALID_MASK;
    }
};

struct MemDetails {
    uint64_t size;
    uint64_t addr;
    uint32_t key;
};

struct IbVerbsData {
    MemDetails remoteInput;
    MemDetails remoteOutput;
    MemDetails localInput;
    MemDetails localOutput;
    uint8_t res[24];
};

struct HcclCombineOpParam {
    uint64_t workSpace;                         // Address for communication between client and server,
                                                // hccl requests and clears
    uint64_t workSpaceSize;                     // Space for communication between client and server
    uint32_t rankId;                            // id of this rank
    uint32_t rankNum;                           // num of ranks in this comm group
    uint64_t winSize;                           // size of each windows memory
    uint64_t windowsIn[HCCL_MAX_RANK_NUM];      // windows address for input, windowsIn[rankId] corresponds
                                                // to the local card address,
                                                // and others are cross-card mapping addresses.
    uint64_t windowsOut[HCCL_MAX_RANK_NUM];     // windows address for output, windowsOut[rankId] corresponds
                                                // to the local card address,
                                                // and others are cross-card mapping addresses.
    uint8_t res[8328];
    uint8_t multiFlag;
    __gm__ IbVerbsData *data;
};
}  // namespace AscendC

#endif  // IMPL_HCCL_HCCL_MSG_H