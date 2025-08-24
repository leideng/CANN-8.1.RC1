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
 * \file kfc_comm_server.h
 * \brief
 */
#ifndef __INTF_KFC_COMM_SERVER_H__
#define __INTF_KFC_COMM_SERVER_H__

#include "kfc_comm.h"

namespace AscendC {
class KfcCommServer {
public:
    __gm__ KfcMsg* msgSendHead;  // Message header
    __gm__ KfcMsg* msgSendStart; // the global position of the initialized message.

    // Receiving Message Queue Maintenance
    __gm__ KfcMsg* msgRcvHead;
    __gm__ KfcMsg* msgRcvStart;

    GM_ADDR ubAvalidTail;

    uint8_t msgSendPos; // for the subBlockID of the AIC core
    uint8_t msgRcvPos;  // for the subBlockID of the AIC core
    uint8_t subBlockID; // for the subBlockID of the AIC core

public:
    __aicore__ inline void Init(GM_ADDR workspace, int i)
    {
        // the Rcv on the server is the same as the Send on the client. The addresses of aic and aiv are swap.
        this->msgRcvStart = (__gm__ KfcMsg*)GetMsgHead(workspace, i);
        this->msgSendStart = this->msgRcvStart + MAX_MSG_COUNT;

        this->msgSendHead = this->msgSendStart;
        this->msgSendPos = 0;
        this->msgRcvHead = this->msgRcvStart;
        this->msgRcvPos = 0;
        this->subBlockID = i;
        ASCENDC_ASSERT((this->msgSendStart != nullptr),
            { KERNEL_LOG(KERNEL_ERROR, "msgSendStart can not be nullptr"); });
        ASCENDC_ASSERT((this->msgRcvStart != nullptr),
            { KERNEL_LOG(KERNEL_ERROR, "msgRcvStart can not be nullptr"); });
        ubAvalidTail = GetUBAvaliedAddr(workspace, i);
    }

    __aicore__ inline __gm__ KfcMsg* AllocMessage()
    {
        return AllocMessageImpl(this->msgSendHead, this->msgSendPos, this->msgSendStart);
    }

    __aicore__ inline void FreeMessage(__gm__ KfcMsg* msg)
    {
        FreeMessageImpl(msg);
    }

    __aicore__ inline void FreeUB(int32_t addr)
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);
        __cbuf__ uint32_t* dst = (__cbuf__ uint32_t*)(TOTAL_L1_SIZE);
#if ASCENDC_CPU_DEBUG
        dst = (uint32_t*)(GetTPipePtr()->GetBaseAddr((uint8_t)(TPosition::A1)) + TOTAL_L1_SIZE);
        *dst = addr;
#else
        create_cbuf_matrix((__cbuf__ uint32_t*)dst, 0x10001, (uint32_t)addr);
#endif
        eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventID);
        WaitFlag<HardEvent::MTE2_MTE3>(eventID);

#ifdef __MSTX_DFX_REPORT__
        MstxCrossRecord record = {
            .addr = reinterpret_cast<uint64_t>(ubAvalidTail),
            .flagId = 1,
            .pipe = pipe_t::PIPE_MTE3,
        };
        __mstx_dfx_report_stub(1, sizeof(MstxCrossRecord), &record);
#endif

        copy_cbuf_to_gm((__gm__ void*)ubAvalidTail, (__cbuf__ void*)dst, 0, 1, 1, 1, 1);
    }

    __aicore__ inline __gm__ KfcMsg* RcvMessage()
    {
        auto msg = (__gm__ KfcMsg*)RcvMessageImpl(this->msgRcvHead, this->msgRcvPos, this->msgRcvStart);
        return msg;
    }

    __aicore__ inline void RollBackMsg()
    {
        RollBackMsgImpl(this->msgRcvHead, this->msgRcvPos);
        return;
    }
};

typedef KfcCommServer* KFC_COMM_SERVER_PTR;
#define KFC_COMM_SERVER KfcCommServer
} // namespace AscendC
#endif // __INTF_KFC_COMM_SERVER_H__
