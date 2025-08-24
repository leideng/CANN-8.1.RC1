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
 * \file hccl_impl_def.h
 * \brief
 */
#ifndef IMPL_HCCL_HCCL_IMPL_DEF_H
#define IMPL_HCCL_HCCL_IMPL_DEF_H
#include "hccl_msg.h"
#include "hccl_control.h"

namespace AscendC {
template <HcclServerType serverType, const auto &config>
class HcclImpl {
public:
    template <bool commit = false>
    __aicore__ inline HcclHandle AllReduce(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t count,
                                           HcclDataType dataType, HcclReduceOp op, uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle AllGather(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t sendCount,
                                           HcclDataType dataType, uint64_t strideCount, uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle ReduceScatter(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t recvCount,
                                               HcclDataType dataType, HcclReduceOp op, uint64_t strideCount,
                                               uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle AlltoAll(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t dataCount,
                                          HcclDataType dataType, uint64_t strideCount = 0, uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle AlltoAllV(GM_ADDR sendBuf, void *sendCounts, void *sdispls, HcclDataType sendType,
                                           GM_ADDR recvBuf, void *recvCounts, void *rdispls, HcclDataType recvType,
                                           uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle BatchWrite(GM_ADDR batchWriteInfo, uint32_t itemNum);

public:
    __aicore__ inline void Init(GM_ADDR context, __gm__ void *initTiling = nullptr);

    __aicore__ inline int32_t SetCcTiling(__gm__ void *ccOpTilingData);

    __aicore__ inline void Commit(HcclHandle handleId);

    __aicore__ inline int32_t Wait(HcclHandle handleId);

    __aicore__ inline int32_t Query(HcclHandle handleId);

    __aicore__ inline void InterHcclGroupSync(int8_t srcGroupID, HcclHandle srcHandleID);

    template <bool sync = true>
    __aicore__ inline int32_t Iterate(HcclHandle handleId, uint16_t *seqSlices, uint16_t seqSliceLen);

    __aicore__ inline void Finalize();

public:
    __aicore__ inline GM_ADDR GetWindowsInAddr(uint32_t rankId);

    __aicore__ inline GM_ADDR GetWindowsOutAddr(uint32_t rankId);

    __aicore__ inline uint32_t GetRankId() { return hcclContext_->rankId; }

    __aicore__ inline uint32_t GetRankDim() { return hcclContext_->rankNum; }

private:
    // Generic implementation for corresponding interface of each Prepare primitive. Return identifier(handleId) of
    // corresponding comm task. HandleId >= 0 when successful, otherwise return -1.
    template <bool commit = false>
    __aicore__ inline HcclHandle CommonPrepareImpl(const CommonPrepareParam &param);

    __aicore__ inline bool CheckCommonPrepareParamValid(const CommonPrepareParam &param);

    // Clear the finishedTurnCnt before aicore exits to ensure the correctness of next launch.
    __aicore__ inline void ResetFinishedTurnCnt();

    __aicore__ inline void SendMsgToServer(const CommonPrepareParam &para,
                                           int8_t srcGroupID = -1, HcclHandle srcHandleID = INVALID_HANDLE_ID);

    __aicore__ inline void SendMsgToServer(const AlltoAllVParamExt &para);

    __aicore__ inline void SetCommitTurnCntToGm(uint8_t msgPos, uint64_t turnCnt);

    __aicore__ inline uint64_t WaitFinishCntFromGm(uint8_t msgPos, uint64_t expectedCnt);

private:
    uint64_t ccOpTilingDataTable_[static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALL)] = {0UL};
    __gm__ HcclCombineOpParam *hcclContext_;
    __gm__ HcclMsgArea *hcclMsgArea_;
    uint8_t handleId2CmdType_[HCCL_MAX_HANDLE_ID] = {0U};
    int8_t handleIdMsgPosition_[HCCL_MAX_HANDLE_ID];
    uint8_t handleIdCommitTurnCnt_[HCCL_MAX_HANDLE_ID] = {0U};
    uint8_t handleIdRepeat_[HCCL_MAX_HANDLE_ID] = {0U};
    uint8_t handleIdWaitCallNum_[HCCL_MAX_HANDLE_ID] = {0U};
    HcclHandle curHandleId_ = INVALID_HANDLE_ID;
    // Current msg position where Api write, starts from 0 and increases automatically, with a maximum of
    // HCCL_MSG_CNT-1. When HCCL_MSG_CNT is reached, take the remainder and recycling message area.
    // Prepare/BatchPrepare/Finalize/InterHcclGroupSync (supported in future versions) only use one message.
    uint8_t curMsgPosition_ = 0U;
    int8_t curVersion_ = -1;
};
}  // namespace AscendC
#endif  // IMPL_HCCL_HCCL_IMPL_DEF_H
