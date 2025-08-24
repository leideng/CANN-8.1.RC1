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
 * \file hccl_v220_impl.h
 * \brief
 */
#ifndef IMPL_HCCL_HCCL_V220_IMPL_H
#define IMPL_HCCL_HCCL_V220_IMPL_H

#include "hccl_common.h"
#include "hccl_control.h"

namespace AscendC {

template <HcclServerType serverType, const auto &config>
__aicore__ inline void HcclImpl<serverType, config>::SendMsgToServer(const CommonPrepareParam &para,
                                                                     int8_t srcGroupID, HcclHandle srcHandleID)
{
    __gm__ HcclMsg *hcclSendMsg = &(hcclMsgArea_->sendMsgs[curMsgPosition_]);
    do {
#if defined(AICORE_EXCEPTION_RESTART) && AICORE_EXCEPTION_RESTART == 1
        if (unlikely(CheckIfRestart(hcclMsgArea_))) {
            return;
        }
#endif
        FlushDataCache(hcclSendMsg);
    } while ((curVersion_ == 0 && hcclSendMsg->addMsg.v0Msg.valid == HCCL_MSG_VALID_MASK) ||
        (curVersion_ != 0 && hcclSendMsg->addMsg.v1Msg.valid == HCCL_MSG_VALID_MASK));
    KERNEL_LOG(KERNEL_INFO, "Hccl send msg[%u] is available now.", curMsgPosition_);
    if (srcGroupID < 0) {
        para.AssembleHcclMsg(curVersion_, curHandleId_,
                             ccOpTilingDataTable_[static_cast<uint32_t>(para.commType)], hcclSendMsg, hcclMsgArea_);
    } else {
        para.AssembleHcclMsg(srcGroupID, srcHandleID, hcclSendMsg);
    }
    FlushDataCache(reinterpret_cast<__gm__ void *>(hcclSendMsg));
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline void HcclImpl<serverType, config>::SendMsgToServer(const AlltoAllVParamExt &para)
{
    __gm__ HcclMsgExt *hcclSendMsg = &(hcclMsgArea_->paramExtMsgList[curMsgPosition_]);
    do {
#if defined(AICORE_EXCEPTION_RESTART) && AICORE_EXCEPTION_RESTART == 1
        if (unlikely(CheckIfRestart(hcclMsgArea_))) {
            return;
        }
#endif
        FlushDataCache(hcclSendMsg);
    } while (hcclSendMsg->valid == HCCL_MSG_VALID_MASK);
    KERNEL_LOG(KERNEL_INFO, "Hccl send extMsg[%u] is available now.", curMsgPosition_);
    para.AssembleHcclMsgExt(hcclContext_->rankNum, hcclSendMsg);
    GlobalTensor<int64_t> globalHcclMsgArea;
    for (uint32_t i = 0U; i < hcclContext_->rankNum; i += U64_CNT_PER_CACHELINE) {
        FlushDataCache(globalHcclMsgArea, (hcclSendMsg->sendCounts + i));
        FlushDataCache(globalHcclMsgArea, (hcclSendMsg->sendOffset + i));
        FlushDataCache(globalHcclMsgArea, (hcclSendMsg->recvCounts + i));
        FlushDataCache(globalHcclMsgArea, (hcclSendMsg->recvOffset + i));
    }
    FlushDataCache(globalHcclMsgArea, hcclSendMsg->reserved);
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline bool HcclImpl<serverType, config>::CheckCommonPrepareParamValid(const CommonPrepareParam &param)
{
    const HcclCMDType commType = param.commType;
    if (curVersion_ > 0) {
        ASCENDC_HCCL_API_ASSERT(ccOpTilingDataTable_[static_cast<uint32_t>(commType)] != 0UL, { return false; },
                                "Failed to prepare for type %u, ensure SetCcTiling has been called.",
                                static_cast<uint32_t>(commType));
    } else {
        ASCENDC_HCCL_API_ASSERT(curVersion_ >= 0, { return false; },
                                "Failed to prepare for type %u, ensure Init has been called",
                                static_cast<uint32_t>(commType));
    }
    ASCENDC_HCCL_API_ASSERT(param.sendBuf != nullptr && param.recvBuf != nullptr,
                            { return false; }, "Call Prepare[%d] failed, the param sendBuf/recvBuf is nullptr, "
                            "which is an invalid parameter.", static_cast<int32_t>(commType));
    ASCENDC_HCCL_API_ASSERT(param.dataType >= HCCL_DATA_TYPE_INT8 &&
                            param.dataType < HCCL_DATA_TYPE_RESERVED, { return false; },
                            "Call Prepare[%d] failed, param HcclDataType is %d, invalid.",
                            static_cast<int32_t>(commType), static_cast<int32_t>(param.dataType));
    if (commType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        ASCENDC_HCCL_API_ASSERT(param.paramExt.sendCounts != nullptr && param.paramExt.sdispls != nullptr &&
                                param.paramExt.recvCounts != nullptr && param.paramExt.rdispls != nullptr,
                                { return false; }, "Call AlltoAllV failed, "
                                                   "param sendCounts/recvCounts/sdispls/rdispls is nullptr, invalid.");
    } else {
        ASCENDC_HCCL_API_ASSERT(param.count != 0, { return false; },
                                "Call Prepare[%d] failed, param sendCount/recvCount is 0, invalid.",
                                static_cast<int32_t>(commType));
    }
    return true;
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline void HcclImpl<serverType, config>::SetCommitTurnCntToGm(uint8_t msgPos, uint64_t turnCnt)
{
    if (GetBlockIdx() != DEFAULT_CFG.blockId) {
        return;
    }

    __gm__ TurnCnt *commitGM = hcclMsgArea_->commitTurnCnt + msgPos;
    do {
#if defined(AICORE_EXCEPTION_RESTART) && AICORE_EXCEPTION_RESTART == 1
        if (unlikely(CheckIfRestart(hcclMsgArea_))) {
            return;
        }
#endif
        FlushDataCache(commitGM);
    } while (commitGM->cnt >= turnCnt);
    KERNEL_LOG(KERNEL_INFO, "Block idx[%d] write commit turn cnt[%lu].", DEFAULT_CFG.blockId, turnCnt);
    commitGM->cnt = turnCnt;
    commitGM->valid = COMMIT_VALID_MASK;
    FlushDataCache(commitGM);
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline uint64_t HcclImpl<serverType, config>::WaitFinishCntFromGm(uint8_t msgPos, uint64_t expectedCnt)
{
    __gm__ TurnCnt *finishGM = hcclMsgArea_->finishedTurnCnt + msgPos;
    GlobalTensor<int64_t> globalHcclMsgArea;
    while (true) {
#if defined(AICORE_EXCEPTION_RESTART) && AICORE_EXCEPTION_RESTART == 1
        if (unlikely(CheckIfRestart(hcclMsgArea_))) {
            break;
        }
#endif
        FlushDataCache(globalHcclMsgArea, finishGM);
        if (finishGM->cnt >= expectedCnt) {
            break;
        }
    }
    return finishGM->cnt;
}

template <HcclServerType serverType, const auto &config>
template <bool commit>
__aicore__ inline HcclHandle HcclImpl<serverType, config>::CommonPrepareImpl(const CommonPrepareParam &param)
{
#if defined(AICORE_EXCEPTION_RESTART) && AICORE_EXCEPTION_RESTART == 1
    if (unlikely(CheckIfRestart(hcclMsgArea_))) {
        KERNEL_LOG(KERNEL_INFO, "Prepare pass, need to restart.");
        return INVALID_HANDLE_ID;
    }
#endif
    if (unlikely(param.repeat == 0U)) {
        return INVALID_HANDLE_ID;
    }
    ASCENDC_HCCL_API_ASSERT(CheckCommonPrepareParamValid(param), { return INVALID_HANDLE_ID; },
                            "Call Prepare[%d] failed, param invalid.",
                            static_cast<int32_t>(param.commType));

    HcclHandle handleId = ++curHandleId_;
    ASCENDC_HCCL_API_ASSERT(handleId < HCCL_MAX_HANDLE_ID, { return INVALID_HANDLE_ID; },
                            "Call Prepare[%d] failed, Prepare interface call num is[%d], expected no more than[%d].",
                            static_cast<int32_t>(param.commType), handleId + 1, HCCL_MAX_HANDLE_ID);

    if (GetBlockIdx() == DEFAULT_CFG.blockId) {
        KERNEL_LOG(KERNEL_INFO, "Only block idx[%ld] write sendMsgList[%u] when prepare[comm type: %d].",
                   GetBlockIdx(), curMsgPosition_, static_cast<int32_t>(param.commType));
        if (param.commType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
            SendMsgToServer(param.paramExt);
        }
        SendMsgToServer(param);
    }

    handleIdMsgPosition_[handleId] = curMsgPosition_;
    handleIdRepeat_[handleId] = param.repeat;
    handleId2CmdType_[handleId] = static_cast<uint8_t>(param.commType);
    if constexpr (commit) {
        handleIdCommitTurnCnt_[handleId] = param.repeat;
        SetCommitTurnCntToGm(curMsgPosition_, handleIdCommitTurnCnt_[handleId]);
    }
    ++curMsgPosition_;
    ASCENDC_HCCL_API_ASSERT(curMsgPosition_ < HCCL_MSG_CNT, {return INVALID_HANDLE_ID; },
                            "Message amount exceeds the maximum value when prepare.");
    return handleId;
}

template <HcclServerType serverType, const auto &config>
template <bool commit>
__aicore__ inline HcclHandle
HcclImpl<serverType, config>::AllReduce(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t count, HcclDataType dataType,
                                        HcclReduceOp op, uint8_t repeat)
{
    ASCENDC_HCCL_API_ASSERT(op >= HCCL_REDUCE_SUM && op < HCCL_REDUCE_RESERVED, { return INVALID_HANDLE_ID; },
                            "Call AllReduce failed, param HcclReduceOp is %d, invalid.", static_cast<int32_t>(op));

    return CommonPrepareImpl<commit>({ HcclCMDType::HCCL_CMD_ALLREDUCE, sendBuf, recvBuf, count, dataType,
                                       op, 0, repeat });
}

template <HcclServerType serverType, const auto &config>
template <bool commit>
__aicore__ inline HcclHandle
HcclImpl<serverType, config>::AllGather(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t sendCount, HcclDataType dataType,
                                        uint64_t strideCount, uint8_t repeat)
{
    return CommonPrepareImpl<commit>({ HcclCMDType::HCCL_CMD_ALLGATHER, sendBuf, recvBuf, sendCount, dataType,
                                       HCCL_REDUCE_RESERVED, strideCount, repeat });
}

template <HcclServerType serverType, const auto &config>
template <bool commit>
__aicore__ inline HcclHandle
HcclImpl<serverType, config>::ReduceScatter(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t recvCount, HcclDataType dataType,
                                            HcclReduceOp op, uint64_t strideCount, uint8_t repeat)
{
    ASCENDC_HCCL_API_ASSERT(op >= HCCL_REDUCE_SUM && op < HCCL_REDUCE_RESERVED, { return INVALID_HANDLE_ID; },
                            "Call ReduceScatter failed, param HcclReduceOp is %d, invalid.", static_cast<int32_t>(op));
    return CommonPrepareImpl<commit>({ HcclCMDType::HCCL_CMD_REDUCE_SCATTER, sendBuf, recvBuf, recvCount,
                                       dataType, op, strideCount, repeat });
}

template <HcclServerType serverType, const auto &config>
template <bool commit>
__aicore__ inline HcclHandle
HcclImpl<serverType, config>::AlltoAll(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t dataCount, HcclDataType dataType,
                                       uint64_t strideCount, uint8_t repeat)
{
    return CommonPrepareImpl<commit>({ HcclCMDType::HCCL_CMD_ALLTOALL, sendBuf, recvBuf, dataCount, dataType,
                                       HCCL_REDUCE_RESERVED, strideCount, repeat });
}

template <HcclServerType serverType, const auto &config>
template <bool commit>
__aicore__ inline HcclHandle
HcclImpl<serverType, config>::AlltoAllV(GM_ADDR sendBuf, void *sendCounts, void *sdispls, HcclDataType sendType,
                                        GM_ADDR recvBuf, void *recvCounts, void *rdispls, HcclDataType recvType,
                                        uint8_t repeat)
{
    ASCENDC_HCCL_API_ASSERT(sendType == recvType, { return INVALID_HANDLE_ID; },
                            "Call AlltoAllV failed, param sendType[%d] is not equal to recvType[%d], invalid.",
                            static_cast<int32_t>(sendType), static_cast<int32_t>(recvType));
    return CommonPrepareImpl<commit>({ HcclCMDType::HCCL_CMD_ALLTOALLV, sendBuf, recvBuf, 0U, sendType,
                                       HCCL_REDUCE_RESERVED, 0U, repeat,
                                       {static_cast<uint64_t *>(sendCounts), static_cast<uint64_t *>(sdispls),
                                        static_cast<uint64_t *>(recvCounts), static_cast<uint64_t *>(rdispls)} });
}

template <HcclServerType serverType, const auto &config>
template <bool commit>
__aicore__ inline HcclHandle HcclImpl<serverType, config>::BatchWrite(GM_ADDR batchWriteInfo, uint32_t itemNum)
{
    return CommonPrepareImpl<true>({HcclCMDType::HCCL_CMD_BATCH_WRITE, batchWriteInfo, batchWriteInfo, itemNum,
                                    HCCL_DATA_TYPE_INT8});
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline void HcclImpl<serverType, config>::Init(GM_ADDR context, __gm__ void *initTiling)
{
    ASCENDC_HCCL_API_ASSERT(context != nullptr, { return; }, "Init Hccl failed, context addr is nullptr.");
    hcclContext_ = (__gm__ HcclCombineOpParam *)context;
    // ensure hcclMsgArea 512B aligned
    uint64_t msgAddr = hcclContext_->workSpace;
    if (msgAddr & 0x1ff) {
        msgAddr = (msgAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    if (unlikely((msgAddr == 0UL) || (initTiling == nullptr && curVersion_ > 0) ||
        (initTiling != nullptr && curVersion_ == 0))) {
        KERNEL_LOG(KERNEL_ERROR, "Init Hccl failed, workspace addr is nullptr or invalid tiling.");
        curVersion_ = -1;
        return;
    }
    curVersion_ = (initTiling != nullptr ? 1 : 0);
    hcclMsgArea_ = (__gm__ HcclMsgArea *)msgAddr;
    for (uint32_t i = 0U; i < HCCL_MAX_HANDLE_ID; ++i) {
        handleIdMsgPosition_[i] = INVALID_MSG_POSITION;
    }
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline int32_t HcclImpl<serverType, config>::SetCcTiling(__gm__ void *ccOpTilingData)
{
    ASCENDC_HCCL_API_ASSERT(curVersion_ == 1, { return HCCL_FAILED; },
                            "Call SetCcTiling failed, ensure Hccl::InitV1 func has been called successfully!");
    ASCENDC_HCCL_API_ASSERT(ccOpTilingData != nullptr, { return HCCL_FAILED; },
                            "Call SetCcTiling failed, ensure ccOpTilingData is not nullptr");
    auto ccTilingPtr = reinterpret_cast<__gm__ char *>(ccOpTilingData);
    auto cmdType = *(reinterpret_cast<__gm__ uint32_t *>(ccTilingPtr + HCCL_CMD_TYPE_OFFSET));
    ASCENDC_HCCL_API_ASSERT(cmdType >= 0 && cmdType < static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALL),
                            { return HCCL_FAILED; }, "Call SetCcTiling failed, ensure cmdType is valid");
    KERNEL_LOG(KERNEL_INFO, "CmdType = %d, ccOpTilingData = %lu ", cmdType, reinterpret_cast<uint64_t>(ccOpTilingData));
    ccOpTilingDataTable_[cmdType] = reinterpret_cast<uint64_t>(ccOpTilingData);
    return HCCL_SUCCESS;
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline int32_t HcclImpl<serverType, config>::Wait(HcclHandle handleId)
{
#if defined(AICORE_EXCEPTION_RESTART) && AICORE_EXCEPTION_RESTART == 1
    if (unlikely(CheckIfRestart(hcclMsgArea_))) {
        KERNEL_LOG(KERNEL_INFO, "Wait pass, need to restart.");
        return HCCL_FAILED;
    }
#endif
    ASCENDC_HCCL_API_ASSERT(curVersion_ >= 0, { return HCCL_FAILED; },
                            "Call Wait failed, ensure Hccl::Init func has been called successfully!");
    if (unlikely(handleId <= INVALID_HANDLE_ID || handleId >= HCCL_MAX_HANDLE_ID)) {
        KERNEL_LOG(KERNEL_ERROR, "Failed to wait, handleId is[%d], expected to be in range of [0, %d).",
                   handleId, HCCL_MAX_HANDLE_ID);
        return HCCL_FAILED;
    }
    uint8_t &waitCnt = handleIdWaitCallNum_[handleId];
    if (unlikely(waitCnt >= handleIdCommitTurnCnt_[handleId])) {
        KERNEL_LOG(KERNEL_ERROR, "Failed to wait, call num of Wait for handleId[%d] is[%u], expected to be no larger "
                                 "than Commit num[%u].", handleId, waitCnt + 1, handleIdCommitTurnCnt_[handleId]);
        return HCCL_FAILED;
    };
    int8_t curMsgPos = handleIdMsgPosition_[handleId];
    ASCENDC_HCCL_API_ASSERT(curMsgPos != INVALID_MSG_POSITION, { return HCCL_FAILED; },
                            "Call Wait failed, handleId[%d] was not got by Prepare interface.", handleId);
    (void)WaitFinishCntFromGm(curMsgPos, ++waitCnt);
    return HCCL_SUCCESS;
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline int32_t HcclImpl<serverType, config>::Query(HcclHandle handleId)
{
    ASCENDC_HCCL_API_ASSERT(curVersion_ >= 0, { return HCCL_FAILED; },
                            "Call Query failed, ensure Hccl::Init func has been called successfully!");
    ASCENDC_HCCL_API_ASSERT((handleId > INVALID_HANDLE_ID) && (handleId < HCCL_MAX_HANDLE_ID), { return HCCL_FAILED; },
                            "Call Query failed, handleId is[%d], expected in range of [0, %d).",
                            handleId, HCCL_MAX_HANDLE_ID);
    int8_t curMsgPos = handleIdMsgPosition_[handleId];
    ASCENDC_HCCL_API_ASSERT(curMsgPos != INVALID_MSG_POSITION, { return HCCL_FAILED; },
                            "Call Query failed, handleId[%d] was not got by Prepare interface.", handleId);
    return WaitFinishCntFromGm(curMsgPos, 0UL);
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline void HcclImpl<serverType, config>::Commit(HcclHandle handleId)
{
#if defined(AICORE_EXCEPTION_RESTART) && AICORE_EXCEPTION_RESTART == 1
    if (unlikely(CheckIfRestart(hcclMsgArea_))) {
        KERNEL_LOG(KERNEL_INFO, "Commit pass, need to restart.");
        return;
    }
#endif
    ASCENDC_HCCL_API_ASSERT(curVersion_ >= 0, { return; },
                            "Call Commit failed, ensure Hccl::Init func has been called successfully!");
    if (unlikely(handleId <= INVALID_HANDLE_ID || handleId >= HCCL_MAX_HANDLE_ID)) {
        KERNEL_LOG(KERNEL_ERROR, "Call Commit failed, handleId is[%d], expected in range of [0, %d).",
                   handleId, HCCL_MAX_HANDLE_ID);
        return;
    }
    uint8_t &commitCnt = handleIdCommitTurnCnt_[handleId];
    if (unlikely(commitCnt >= handleIdRepeat_[handleId])) {
        KERNEL_LOG(KERNEL_ERROR, "Call Commit for handleId[%d] failed, call num is[%u], "
                                 "expected no larger than task num[%u].", handleId, commitCnt + 1,
                                 handleIdRepeat_[handleId]);
        return;
    }
    SetCommitTurnCntToGm(handleIdMsgPosition_[handleId], ++commitCnt);
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline void HcclImpl<serverType, config>::InterHcclGroupSync(int8_t srcGroupID, HcclHandle srcHandleID)
{
    ASCENDC_HCCL_API_ASSERT(curVersion_ >= 0, { return; },
                            "Call InterHcclGroupSync failed, ensure Hccl::Init func has been called successfully!");

    if (GetBlockIdx() == DEFAULT_CFG.blockId) {
        KERNEL_LOG(KERNEL_INFO, "Only block idx[%ld] write sendMsgList[%u] when InterHcclGroupSync.",
                   GetBlockIdx(), curMsgPosition_);
        CommonPrepareParam param = {HcclCMDType::HCCL_CMD_INTER_GROUP_SYNC};
        SendMsgToServer(param, srcGroupID, srcHandleID);
    }
    ++curMsgPosition_;
    ASCENDC_HCCL_API_ASSERT(curMsgPosition_ < HCCL_MSG_CNT, {return; },
                            "Message amount exceeds the maximum value when sync group.");
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline void HcclImpl<serverType, config>::Finalize()
{
#if defined(AICORE_EXCEPTION_RESTART) && AICORE_EXCEPTION_RESTART == 1
    if (unlikely(CheckIfRestart(hcclMsgArea_))) {
        KERNEL_LOG(KERNEL_INFO, "Finalize pass, need to restart.");
        return;
    }
#endif
    ASCENDC_HCCL_API_ASSERT(curVersion_ >= 0, { return; },
                            "Call Finalize failed, ensure Hccl::Init func has been called successfully!");

    if (GetBlockIdx() == DEFAULT_CFG.blockId) {
        // 1. wait until last hccl task finished(the commitTurnCnt will be reset by aicpu-server before task finished),
        //    then commitTurnCnt can be used by next op.
        if (curHandleId_ > INVALID_HANDLE_ID) {
            KERNEL_LOG(KERNEL_INFO, "Wait hccl task finished for last HandleId[%d] when Finalize.", curHandleId_);
            while (Query(curHandleId_) < handleIdRepeat_[curHandleId_]) {
#if defined(AICORE_EXCEPTION_RESTART) && AICORE_EXCEPTION_RESTART == 1
                if (unlikely(CheckIfRestart(hcclMsgArea_))) {
                    return;
                }
#endif
            }
        }
        // 2. send Finalize msg
        KERNEL_LOG(KERNEL_INFO, "Only block idx[%ld] write sendMsgList[%u] when Finalize.",
                   GetBlockIdx(), curMsgPosition_);
        CommonPrepareParam param = {HcclCMDType::HCCL_CMD_FINALIZE};
        SendMsgToServer(param);
        // 3. wait until Finalize msg has been read, then the prepare msg area can be used by next op.
        KERNEL_LOG(KERNEL_INFO, "Only block idx[%ld] wait until Finalize msg has been read.", GetBlockIdx());
        // 4. wait for server sqe task finished, and client can ResetFinishedTurnCnt
        __gm__ TurnCnt *finishGM = hcclMsgArea_->finishedTurnCnt + curMsgPosition_;
        KERNEL_LOG(KERNEL_INFO, "Only block idx[%ld] wait for server sqe task finished.", GetBlockIdx());
        do {
#if defined(AICORE_EXCEPTION_RESTART) && AICORE_EXCEPTION_RESTART == 1
            if (unlikely(CheckIfRestart(hcclMsgArea_))) {
                return;
            }
#endif
            FlushDataCache(finishGM);
        } while (finishGM->cnt != FINALIZE_FINISH_CNT);
        // 5. reset finishedTurnCnt, then the finishedTurnCnt can be used by next op.
        KERNEL_LOG(KERNEL_INFO, "Only block idx[%ld] will ResetFinishedTurnCnt.", GetBlockIdx());
        ResetFinishedTurnCnt();
    }
    ++curMsgPosition_;
    ASCENDC_HCCL_API_ASSERT(curMsgPosition_ < HCCL_MSG_CNT, {return; },
                            "Message amount exceeds the maximum value when finalize.");
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline GM_ADDR HcclImpl<serverType, config>::GetWindowsInAddr(uint32_t rankId)
{
    ASCENDC_HCCL_API_ASSERT(rankId < GetRankDim(), { return nullptr; },
                            "GetWindowsInAddr failed, rankId[%u], expected less than[%u]", rankId, GetRankDim());
    if (hcclContext_->multiFlag == 0U) {
        return (GM_ADDR)hcclContext_->windowsIn[rankId];
    } else {
        if (rankId == hcclContext_->rankId) {
            return (GM_ADDR)(hcclContext_->data[rankId].localInput.addr);
        } else {
            return (GM_ADDR)(hcclContext_->data[rankId].remoteInput.addr);
        }
    }
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline GM_ADDR HcclImpl<serverType, config>::GetWindowsOutAddr(uint32_t rankId)
{
    ASCENDC_HCCL_API_ASSERT(rankId < GetRankDim(), { return nullptr; },
                            "GetWindowsOutAddr failed, rankId[%u], expected less than[%u]", rankId, GetRankDim());
    if (hcclContext_->multiFlag == 0U) {
        return (GM_ADDR)hcclContext_->windowsOut[rankId];
    } else {
        if (rankId == hcclContext_->rankId) {
            return (GM_ADDR)(hcclContext_->data[rankId].localOutput.addr);
        } else {
            return (GM_ADDR)(hcclContext_->data[rankId].remoteOutput.addr);
        }
    }
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline void HcclImpl<serverType, config>::ResetFinishedTurnCnt()
{
    __gm__ TurnCnt *finishArea = hcclMsgArea_->finishedTurnCnt;
    GlobalTensor<int64_t> globalHcclMsgArea;
    for (uint32_t i = 0U; i <= curMsgPosition_; ++i) {
        __gm__ TurnCnt *finishGM = finishArea + i;
        finishGM->cnt = 0;
        FlushDataCache(globalHcclMsgArea, finishGM);
    }
}
}  // namespace AscendC
#endif  // __CCE_AICORE__ == 220