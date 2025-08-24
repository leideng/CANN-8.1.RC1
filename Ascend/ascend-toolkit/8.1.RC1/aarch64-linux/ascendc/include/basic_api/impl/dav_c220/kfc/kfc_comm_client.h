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
 * \file kfc_comm_client.h
 * \brief
 */
#ifndef __KERNEL_KFC_COMM_CLIENT_H__
#define __KERNEL_KFC_COMM_CLIENT_H__

#include "kfc_comm.h"

namespace AscendC {
class KfcCommClient {
public:
    // Send Message Queue Maintenance
    __gm__ KfcMsg *msgSendHead;   // Message header
    __gm__ KfcMsg *msgSendStart;  // the global position of the initialized message.

    // Receiving Message Queue Maintenance
    __gm__ KfcMsg *msgRcvHead;
    __gm__ KfcMsg *msgRcvStart;

    GM_ADDR ubStart;
    GM_ADDR ubAvalidTail;

    __ubuf__ KfcMsg *ubMsg;
    uint32_t head;
    uint32_t tail;
    uint8_t msgRcvPos;
    uint8_t msgSendPos;  // Index used for circular queues. msgQueueHead = msgQueueStart + msgPos
    uint8_t eventID_;
    uint8_t enableHardWare; 

public:
    __aicore__ inline KfcCommClient(GM_ADDR workspace, int subBlockID, uint8_t enableHardWare = 0)
    {
        if ASCEND_IS_AIV {
            this->enableHardWare = enableHardWare;
            if (enableHardWare) {
                return;
            }
            ASCENDC_ASSERT((workspace != nullptr), { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
            ASCENDC_ASSERT((GetTPipePtr() != nullptr), { KERNEL_LOG(KERNEL_ERROR, "tpipe ptr can not be nullptr"); });
            ASCENDC_ASSERT((GetTPipePtr()->GetBaseAddr((int8_t)TPosition::VECIN) != nullptr),
                           { KERNEL_LOG(KERNEL_ERROR, "vecin base addr can not be nullptr"); });
            // Note that the addresses of aic and aiv are exchanged.
            this->msgSendStart = (__gm__ KfcMsg *)GetMsgHead(workspace, subBlockID);
            this->msgRcvStart = this->msgSendStart + MAX_MSG_COUNT;

            this->msgSendHead = this->msgSendStart;
            this->msgSendPos = 0;
            this->msgRcvHead = this->msgRcvStart;
            this->msgRcvPos = 0;

            // During debugging, CPU need to know the global variable address of the tpipe.
#if ASCENDC_CPU_DEBUG
            ubMsg = reinterpret_cast<__ubuf__ KfcMsg *>(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::VECIN) +
                TOTAL_UB_SIZE - sizeof(KfcMsg));
#else
            ubMsg = reinterpret_cast<__ubuf__ KfcMsg *>(TOTAL_UB_SIZE - sizeof(KfcMsg));
#endif
            eventID_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_S>();
            SetFlag<HardEvent::MTE3_S>((event_t)eventID_);

            ubStart = GetUBMapAddr(workspace);
            ubAvalidTail = GetUBAvaliedAddr(workspace);
            head = 0;
            tail = 0;
        }
    }

    __aicore__ inline ~KfcCommClient()
    {
        if ASCEND_IS_AIV {
            if (this->enableHardWare) {
                return;
            }
            __gm__ KfcMsg *msg = AllocMessage();
            ASCENDC_ASSERT((msg != nullptr),
                           { KERNEL_LOG(KERNEL_ERROR, "ret of alloc message can not be nullptr when client quit"); });
            uint32_t quitSignal = KfcMsgMakeFlag(KFC_Enum::SERVICE_QUIT, 0);
            *((__gm__ uint32_t *)msg) = quitSignal;

#ifdef __MSTX_DFX_REPORT__
            MstxCrossRecord record = {
                .addr = reinterpret_cast<uint64_t>(msg),
                .flagId = 0,
                .pipe = pipe_t::PIPE_S,
            };
            __mstx_dfx_report_stub(0, sizeof(MstxCrossRecord), &record);
#endif

            dcci(reinterpret_cast<__gm__ int64_t *>(msg), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        }
    }

    template <bool isAck>
    __aicore__ inline void PostMessage(__gm__ KfcMsg *msg)
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventID);
        WaitFlag<HardEvent::S_MTE3>(eventID);
        PipeBarrier<PIPE_MTE3>();
        copy_ubuf_to_gm((__gm__ void *)msg, (__ubuf__ void *)ubMsg, 0, 1, sizeof(KfcMsg) / ONE_BLK_SIZE, 0, 0);

#ifdef __MSTX_DFX_REPORT__
        MstxCrossRecord record = {
            .addr = reinterpret_cast<uint64_t>(msg),
            .flagId = 0,
            .pipe = pipe_t::PIPE_MTE3,
        };
        __mstx_dfx_report_stub(0, sizeof(MstxCrossRecord), &record);
#endif

        SetFlag<HardEvent::MTE3_S>((event_t)this->eventID_);
    }

    __aicore__ inline __gm__ KfcMsg *AllocMessage()
    {
        auto ret = AllocMessageImpl(this->msgSendHead, this->msgSendPos, this->msgSendStart);
        WaitFlag<HardEvent::MTE3_S>((event_t)this->eventID_);
        ASCENDC_ASSERT((ret),
            { KERNEL_LOG(KERNEL_ERROR, "ret of alloc message can not be nullptr"); });
        return ret;
    }

    __aicore__ inline void FreeMessage(__gm__ KfcMsg *msg)
    {
        FreeMessageImpl(msg);
    }

    __aicore__ inline GM_ADDR AllocUB(uint32_t size, int32_t &tailInfo)
    {
#ifdef __MSTX_DFX_REPORT__
        MstxCrossRecord record = {
            .addr = reinterpret_cast<uint64_t>(ubAvalidTail),
            .flagId = 1,
            .pipe = pipe_t::PIPE_S,
            .isMerge = true,
        };
        __mstx_dfx_report_stub(0, sizeof(MstxCrossRecord), &record);
#endif

        GM_ADDR ret;
        if (head + size >= WORKSPACE_UB_SIZE) {
            dcci(reinterpret_cast<__gm__ int64_t *>(ubAvalidTail), cache_line_t::SINGLE_CACHE_LINE,
                dcci_dst_t::CACHELINE_OUT);
            tail = *(reinterpret_cast<__gm__ uint32_t *>(ubAvalidTail));
            while (head < tail || tail == 0) {
                Barrier();
                dcci(reinterpret_cast<__gm__ int64_t *>(ubAvalidTail), cache_line_t::SINGLE_CACHE_LINE,
                    dcci_dst_t::CACHELINE_OUT);
                Barrier();
                tail = *(reinterpret_cast<__gm__ uint32_t *>(ubAvalidTail));
            }
            if (tail == head && size == tail) {
                tail = 0;
            }
            head = 0;
        }

        while (head < tail && (head + size >= tail)) {
            Barrier();
            dcci(reinterpret_cast<__gm__ int64_t *>(ubAvalidTail), cache_line_t::SINGLE_CACHE_LINE,
                dcci_dst_t::CACHELINE_OUT);
            Barrier();
            tail = *(reinterpret_cast<__gm__ uint32_t *>(ubAvalidTail));
        }

#ifdef __MSTX_DFX_REPORT__
        __mstx_dfx_report_stub(0, sizeof(MstxCrossRecord), &record);
#endif

        ret = ubStart + head;
        head += size;
        tailInfo = head;
        return ret;
    }

    __aicore__ inline __gm__ KfcMsg *RcvMessage()
    {
        auto ret = RcvMessageImpl(this->msgRcvHead, this->msgRcvPos, this->msgRcvStart);
        return ret;
    }
};


#if __CCE_AICORE__ == 220
#ifdef __DAV_C220_CUBE__
#elif defined(__DAV_C220_VEC__)
__BLOCK_LOCAL__ __inline__ AscendC::KfcCommClient* g_kfcClient;
#else
__BLOCK_LOCAL__ __inline__ AscendC::KfcCommClient* g_kfcClient;
#endif
#endif

__aicore__ inline AscendC::KfcCommClient* GetKfcClient()
{
#if __CCE_AICORE__ == 220
#ifndef __DAV_C220_CUBE__
    return reinterpret_cast<AscendC::KfcCommClient*>(g_kfcClient);
#else
    return nullptr;
#endif
#else
    ASSERT(g_coreType == AscendC::AIV && "not supported on current device");
    return nullptr;
#endif
}
}  // namespace AscendC
#endif  // __KERNEL_KFC_COMM_CLIENT_H__
