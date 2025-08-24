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
 * \file kfc_comm.h
 * \brief
 */
#ifndef __KERNEL_KFC_COMM_H__
#define __KERNEL_KFC_COMM_H__

#include "kernel_common.h"
#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_common_impl.h"
#include "dav_c100/kernel_operator_set_atomic_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_common_impl.h"
#include "dav_m200/kernel_operator_set_atomic_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_common_impl.h"
#include "dav_c220/kernel_operator_set_atomic_impl.h"
#endif
#include "kfc_log.h"

namespace AscendC {
#define MSG_POS __gm__

__aicore__ inline void Barrier()
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    __asm__ __volatile__("" ::: "memory");
#else
    __asm__ __volatile__("");
#endif
}

enum class KFC_Enum : uint16_t {
    SERVICE_ID_MASK = 0xFF00,
    SERVICE_ID_SCM = 0x0100,
    SCMFUN_GM2L1,
    SCMFUN_GM2L1ND2NZ,
    SERVICE_ID_MATMUL = 0x0300,
    MMFUN_MASK = 0x0380,     // Indicates that Iterate needs to be invoked.
    MMFUN_ITERATE = 0x0380,  // Iterate and IterateAll occupy 2 bits to save the cost of judgment.
    MMFUN_ITERATE_ALL = 0x0381,
    MMFUN_INIT = 0x0301,
    MMFUN_GET_TENSOR_C,
    MMFUN_ITERATE_ALL_RESP,
    MMFUN_GET_TENSOR_C_RESP,
    MMFUN_GET_OFFSET_C,
    MMFUN_GET_OFFSET_C_RESP,
    MMFUN_SET_ORG_SHAPE,
    MMFUN_SET_HF32,
    MMFUN_SET_USER_DEF_INFO,
    MMFUN_ITERATE_BATCH_ALL,
    MMFUN_ITERATE_BATCH_ALL_RESP,
    MMFUN_ITERATE_N_BATCH_ALL,
    MMFUN_ITERATE_N_BATCH_ALL_RESP,
    MMFUN_END,
    CONVFUNC_ITERATE,
    CONVFUNC_ITERATE_ALL,
    CONVFUNC_GET_TENSOR_C,
    CONVFUNC_END,
    SERVICE_QUIT = 0xfd00,
    SERVICE_BALANCE = 0xfe00,
    SERVICE_ID_NONE = 0xff00
};

enum class MSG_STATE : uint8_t {
    STATE_INVALID,
    STATE_SET,
};

// The message length is 128 bytes and must be 512 bytes aligned.
// Therefore, MAX_MSG_COUNT must be an integer multiple of 8.
#ifdef __MIX_CORE_AIC_RATION__
constexpr int32_t MIX_NUM = __MIX_CORE_AIC_RATION__;
#else
constexpr int32_t MIX_NUM = 2;
#endif
constexpr int32_t MIX_COEFFICIENT = 1;
constexpr int32_t MAX_MATMUL_OBJ = 8;
constexpr int MAX_AIV_NUM = 50;
constexpr int MAX_AIC_NUM = 25;
constexpr int ALIGN_SIZE = 32;
constexpr int BIDIRECTION_NUM = 2;
constexpr bool KFC_APPLY_MSG = true;
constexpr uint64_t INC_PROCESS_CHECK = 14;
constexpr uint64_t WORKSPACE_UB_SIZE = TOTAL_UB_SIZE;
constexpr int32_t MAX_GROUP_ID = 32;
constexpr int32_t MM_CNT_MAX = 1024;
constexpr int32_t QUIT_CNT = 4;
constexpr int32_t MAX_SYNC_COUNT = 100000000;
constexpr int32_t MMCNT_L1_RESERVERD_SIZE = 64; // Reserved L1 64B for MMCNT Query
constexpr uint16_t KFC_SYNC_ID = 15;

struct TilingInfo {
    GM_ADDR tilingAddr;  // the GM address of the tiling.
};

struct MatmulOrgShape {
    uint32_t orgM;
    uint32_t orgN;
    uint32_t orgKa;
    uint32_t orgKb;
    uint32_t orgKc;
};

#ifdef __MSTX_DFX_REPORT__
struct MstxCrossRecord {
    uint64_t addr;
    uint16_t flagId;
    pipe_t pipe;
    bool isMore;
    bool isMerge;
};
#endif

struct MatmulConfigParams {
    // offset->8
    uint32_t enAtomic : 8;
    uint32_t enSequentialWrite : 1;
    uint32_t isTransA : 1;
    uint32_t isTransB : 1;
    uint32_t enPartialSum : 1;
    uint32_t setTail : 1;
    uint32_t setTensorA : 1;
    uint32_t setTensorB : 1;
    uint32_t setTensorBias : 1;
    uint32_t setClearBias : 1;
    uint32_t cIsTscm : 1;
    uint32_t isFirstIter : 1;
    uint32_t sync : 1;
    uint32_t enHF32 : 1;
    uint32_t hf32TransMode : 1;
    uint32_t setQuant : 1;
    uint32_t setBatch : 1;
    uint32_t waitIterateAll : 1;
    uint32_t waitIterateBatch : 1;
    uint32_t iterateFakeMsg : 1;

    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleK;
    uint32_t sizeAmatrix;
    uint32_t sizeBmatrix;

    uint64_t aAddr;
    uint64_t bAddr;
    uint64_t cAddr;
    uint64_t biasAddr;  // 56 bytes, but access to the maximum message size (64 bytes)
    uint64_t quantAddr; // 64 bytes
    uint32_t quantSize;
    uint32_t quantMode;
    uint64_t quantScalar;        // 80 byets
    uint32_t batchA;
    uint32_t batchB;
    uint32_t matrixStrideA;
    uint32_t matrixStrideB;
    uint32_t matrixStrideC;
    uint32_t batchLoop;
    uint32_t counterId; // query mm task process
    uint32_t reserved0;
    uint64_t dataPtr; // 120 bytes
};

struct Conv3DBpInputConfigParams {
    uint32_t enAtomic : 8;
    uint32_t enSequentialWrite : 1;
    uint32_t setTensorWeight : 1;
    uint32_t sync : 1;
    uint32_t isFirstIter : 1;
    uint32_t setTensorFmap : 1;
    uint32_t setTensorOutBackprop : 1;
    uint32_t setSingleShape : 1;
    uint32_t setStartIdx : 1;
    uint32_t enPartialSum : 1;

    uint32_t singleShapeN;
    uint32_t singleShapeD;
    uint32_t curDinStartIdx;

    uint32_t singleShapeM;
    uint32_t singleShapeK;

    uint64_t weightAddr;
    uint64_t outBackpropAddr;
    uint64_t outputAddr;

    int32_t curHoStartIdx;
    uint32_t res;
    uint32_t res1[16];
};

struct Conv3DBpFilterConfigParams {
    uint32_t enAtomic : 8;
    uint32_t enSequentialWrite : 1;
    uint32_t setTensorFmap : 1;
    uint32_t setTensorOutBackprop : 1;
    uint32_t setSingleShape : 1;
    uint32_t sync : 1;
    uint32_t isFirstIter : 1;
    uint32_t waitIterateAll : 1;
    uint32_t enPartialSum : 1;

    uint32_t singleShapeM;
    uint32_t singleShapeN;
    uint32_t singleShapeK;

    uint64_t fmapAddr;
    uint64_t outBackpropAddr;
    uint64_t outputAddr;

    uint32_t curHoStartIdx;
    uint32_t fmapSize;
    uint32_t outBackpropSize;
    uint32_t res;

    uint32_t res1[16];
};

struct Conv3DForwardConfigParams {
    uint32_t enAtomic: 8; //for enable atomic add
    uint32_t enSequentialWrite: 1;
    uint32_t enSetTensorFmap: 1; //for enbale SetTensorFmap intf
    uint32_t enSetTensorWeight: 1; //for enbale SetTensorWeight intf
    uint32_t enSetTensorBias: 1; //for enbale SetTensorBias intf
    uint32_t sync: 1; //for control sync aic/aiv
    uint32_t enSetSingleOutputShape: 1; //for enbale SetSingleOutputShape intf
    uint32_t enSetFmapStartPosition: 1; //for enbale SetFmapStartPosition intf
    uint32_t waitIterateAll: 1;
    uint32_t enPartialSum: 1; //for part of results of L0C write to gm
    uint32_t fmapSize;

    uint64_t fmapAddr;
    uint64_t weightAddr;
    uint64_t biasAddr;
    uint64_t outputAddr;

    uint32_t weightSize;
    uint32_t biasSize;
    uint32_t singleCoreBatch;
    uint32_t singleCo;
    uint32_t singleDo;
    uint32_t singleCoreM;
    uint32_t singleGroupOpt;
    uint32_t diStartPos;
    uint32_t mStartPos;
    uint32_t ciStartPos;
    uint32_t res[10]; //Reserved，for 120Byte ailgn
};

struct MatmulUserDefInfo {
    uint64_t tilingPtr;
};

constexpr uint16_t KFC_MSG_BYTE_OFFSET = 16;
// AIC->AIV, set the event count.
__aicore__ inline uint16_t KfcMsgGetEvtCnt(uint32_t flag)
{
    return flag & 0x00007fff;
}
// AIV->AIC, set the instance ID
__aicore__ inline uint16_t KfcMsgGetInstID(uint32_t flag)
{
    return flag & 0x000000ff;
}
__aicore__ inline KFC_Enum KfcMsgGetFunID(uint32_t flag)
{
    return static_cast<KFC_Enum>((flag & 0xffff0000) >> KFC_MSG_BYTE_OFFSET);
}
__aicore__ inline uint32_t KfcMsgGetState(uint32_t flag)
{
    return (flag & 0x00008000);
}
__aicore__ inline uint32_t KfcMsgMakeFlag(KFC_Enum funID, uint16_t instID)
{
    return (((static_cast<uint16_t>(funID) << KFC_MSG_BYTE_OFFSET) + 0x8000) + (instID));
}

// Currently, the maximum message size is 64 bytes, which is the same as the size of a CacheLine.
struct KfcMsg {
    uint32_t head = 0;
    int32_t ubAddr = -1;
    union {
        uint8_t buffer[120];  // Used for customizing and formatting messages
        TilingInfo tilingInfo;
        MatmulConfigParams body;
        MatmulOrgShape orgShape;
        MatmulUserDefInfo userDefInfo;
        Conv3DBpInputConfigParams convBpInputBody;
        Conv3DBpFilterConfigParams convBpFilterBody;
        Conv3DForwardConfigParams convForwardBody;
    };
};
struct MsgUBAvalied {
    uint32_t head;
    uint32_t res;
    uint8_t buffer[56];
};
struct MsgMatmulCnt {
    uint32_t head;
    uint32_t res;
    uint8_t buffer[56];
};

struct QuitCnt {
    int32_t head;
    uint32_t res;
    uint8_t buffer[56];
};

struct MmTaskCnt {
    uint32_t head;
    uint32_t res;
    uint8_t buffer[56];
};

struct MsgGroupSync {
    int32_t syncCount; // current sync aiv number
    uint8_t res[28];
    uint32_t allNumber; // aiv number in one group
    uint8_t buffer[28];
};

struct MsgGroupSyncAux {
    // for reliable and dfx
    int32_t curNumber; // number of aiv has been in group
    uint8_t res[28];
    uint32_t idField; // curBlockID << 16 + groupID
    uint8_t buffer[28];
};

__aicore__ inline constexpr int AlignTo32(int size)
{
    return (size + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
}

struct SysWorkspaceDesc {
    KfcMsg kfcMsg[MAX_AIV_NUM * BIDIRECTION_NUM * MAX_MSG_COUNT * MIX_COEFFICIENT];
    MsgMatmulCnt cntMsg[MAX_AIV_NUM * MIX_COEFFICIENT][MAX_MATMUL_OBJ];
    MsgUBAvalied ubMsg[MAX_AIV_NUM];
    uint8_t ubMap[MAX_AIV_NUM][WORKSPACE_UB_SIZE];
    QuitCnt quitCnt[QUIT_CNT];
    MmTaskCnt mmTaskCnt[MM_CNT_MAX];
    MsgGroupSync groupSyncMsg[MAX_GROUP_ID];
    MsgGroupSyncAux groupSyncAuxMsg[MAX_GROUP_ID];
};

__aicore__ inline void ClearWorkspaceImpl(__gm__ uint8_t* workspace)
{
    constexpr uint32_t size = BIDIRECTION_NUM * MAX_MSG_COUNT * AlignTo32(sizeof(KfcMsg)) * MIX_NUM;
    constexpr uint32_t sizeUbmsg = MIX_NUM * AlignTo32(sizeof(MsgUBAvalied));
    constexpr uint32_t offsetUbMsg = MAX_AIV_NUM * BIDIRECTION_NUM * MAX_MSG_COUNT *
        MIX_COEFFICIENT * AlignTo32(sizeof(KfcMsg)) + MAX_AIV_NUM * MIX_COEFFICIENT *
        MAX_MATMUL_OBJ * AlignTo32(sizeof(MsgMatmulCnt));
    constexpr uint32_t block = size / 2048;
    __gm__ uint8_t* msgStartAddr = (__gm__ uint8_t*)(workspace + size * GetBlockIdxImpl());
    __gm__ uint8_t* ubMsgStartAddr = (__gm__ uint8_t*)(workspace + offsetUbMsg + sizeUbmsg * GetBlockIdxImpl());
    create_cbuf_matrix((__cbuf__ uint32_t*)(0), 0x10040, 0);
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    for (size_t i = 0; i < block; i++) {
        copy_cbuf_to_gm((__gm__ void*)(msgStartAddr), (__cbuf__ void*)(0), 0, 1, 64, 1, 1);
        msgStartAddr += 2048;
    }
    copy_cbuf_to_gm((__gm__ void*)(ubMsgStartAddr), (__cbuf__ void*)(0), 0, 1, sizeUbmsg / 32, 1, 1);
    PipeBarrier<PIPE_ALL>();
}
__aicore__ inline GM_ADDR GetMsgHead(GM_ADDR workspace, int i)
{
    // Get the message header of the MSG based on the block/
    // MSG arrangement       AIV                    AIV
    //  MSG0 AIV0-->AIC0   GetBlockIdx()=0,  GetSubBlockIdx()=0,
    //  MSG1 AIV0<--AIC0   GetBlockIdx()=0,  GetSubBlockIdx()=0,
    //  MSG2 AIV1-->AIC0   GetBlockIdx()=0,  GetSubBlockIdx()=1,
    //  MSG3 AIV1<--AIC0   GetBlockIdx()=0,  GetSubBlockIdx()=1,

    ASCENDC_ASSERT((i >= 0 && i < MIX_NUM),
                   { KERNEL_LOG(KERNEL_ERROR, "input i is %d, which should be in range [0, %d)", i, MIX_NUM); });
    ASCENDC_ASSERT((workspace != nullptr),
                   { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });

    auto ptr = reinterpret_cast<__gm__ struct SysWorkspaceDesc *>(workspace);
    if ASCEND_IS_AIV {
        auto flatBlockID = GetBlockIdxImpl();
        return reinterpret_cast<GM_ADDR>(&ptr->kfcMsg[flatBlockID * BIDIRECTION_NUM * MAX_MSG_COUNT]);
    } else {
        // Initialize based on the input i value on AIC.
        auto flatBlockID = (GetBlockIdxImpl() * MIX_NUM + i);
        return reinterpret_cast<GM_ADDR>(&ptr->kfcMsg[flatBlockID * BIDIRECTION_NUM * MAX_MSG_COUNT]);
    }
}

__aicore__ inline GM_ADDR GetUBMapAddr(GM_ADDR workspace, int i = 0)
{
    // Retrieve the corresponding UB mapped address
    // { 48*2* max_size * sizeof(msg); } after MSG
    // UBMap   address arrangement
    //  AIV0   0~256K
    //  AIV1   256K ~ 512K
    //  ...
    ASCENDC_ASSERT((workspace != nullptr),
                   { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });

    uint64_t flatBlockID;
    if ASCEND_IS_AIC {
        flatBlockID = (GetBlockIdxImpl() * MIX_NUM + i);
    } else {
        flatBlockID = GetBlockIdxImpl();
    }

    auto ptr = reinterpret_cast<__gm__ struct SysWorkspaceDesc *>(workspace);
    return reinterpret_cast<GM_ADDR>(ptr->ubMap[flatBlockID]);
}

__aicore__ inline GM_ADDR GetMatmulIncAddr(GM_ADDR workspace, uint32_t flatBlockID, uint32_t instID)
{
    // Maximum MAX_MATMUL_OBJ Matmul objects can be created
    ASCENDC_ASSERT((instID < MAX_MATMUL_OBJ),
                   { KERNEL_LOG(KERNEL_ERROR, "instID i is %d, max matmul num is %d", instID, MAX_MATMUL_OBJ); });
    ASCENDC_ASSERT((workspace != nullptr),
                   { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
    auto ptr = reinterpret_cast<__gm__ struct SysWorkspaceDesc *>(workspace);
    return reinterpret_cast<GM_ADDR>(&(ptr->cntMsg[flatBlockID][instID]));
}

__aicore__ inline GM_ADDR GetUBAvaliedAddr(GM_ADDR workspace, uint32_t i = 0)
{
    ASSERT(workspace != nullptr);
    auto flatBlockID = GetBlockIdxImpl();
    if ASCEND_IS_AIC {
        flatBlockID = flatBlockID * MIX_NUM + i;
    }
    auto ptr = reinterpret_cast<__gm__ struct SysWorkspaceDesc *>(workspace);
    return reinterpret_cast<GM_ADDR>(&(ptr->ubMsg[flatBlockID]));
}

__aicore__ inline __gm__ KfcMsg *AllocMessageImpl(
    __gm__ KfcMsg *&msgSendHead, uint8_t &msgSendPos, __gm__ KfcMsg *&msgSendStart)
{
    auto msg = msgSendHead;
    ASCENDC_ASSERT((msgSendHead != nullptr),
                   { KERNEL_LOG(KERNEL_ERROR, "msgSendHead can not be nullptr"); });
    ASCENDC_ASSERT((msgSendStart != nullptr),
                   { KERNEL_LOG(KERNEL_ERROR, "msgSendStart can not be nullptr"); });
    if constexpr (KFC_APPLY_MSG) {
        dcci(reinterpret_cast<__gm__ int64_t *>(msg), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        while (static_cast<bool>(KfcMsgGetState(msg->head))) {
            Barrier();
            dcci(reinterpret_cast<__gm__ int64_t *>(msg), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
            Barrier();
        }
    }
    ASCENDC_ASSERT((!KfcMsgGetState(msg->head)),
                   { KERNEL_LOG(KERNEL_ERROR, "message state is %u, which should be 0", KfcMsgGetState(msg->head)); });
    msgSendPos++;
    if (msgSendPos >= MAX_MSG_COUNT) {
        msgSendPos = 0;
        msgSendHead = msgSendStart;
    } else {
        msgSendHead++;
    }
    return msg;
}

__aicore__ inline __gm__ KfcMsg *RcvMessageImpl(
    __gm__ KfcMsg *&msgRcvHead, uint8_t &msgRcvPos, __gm__ KfcMsg *&msgRcvStart)
{
    // The size of a single cacheline is 64 bytes.
    // only a single cacheline needs to be updated because runtime can ensure message alignment
    ASCENDC_ASSERT((msgRcvHead != nullptr),
                   { KERNEL_LOG(KERNEL_ERROR, "msgRcvHead can not be nullptr"); });
    ASCENDC_ASSERT((msgRcvStart != nullptr),
                   { KERNEL_LOG(KERNEL_ERROR, "msgRcvStart can not be nullptr"); });
    ASCENDC_ASSERT((msgRcvPos >= 0 && msgRcvPos <= MAX_MSG_COUNT), {
        KERNEL_LOG(KERNEL_ERROR, "msgRcvPos is %d, which should be in range of [0, %d]", msgRcvPos, MAX_MSG_COUNT);
    });
    __gm__ KfcMsg* msg = msgRcvHead;
    Barrier();
    dcci(reinterpret_cast<__gm__ int64_t *>(msg), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    Barrier();

    dc_preload((__gm__ uint64_t*)msg, int64_t(0));

#ifdef __MSTX_DFX_REPORT__
    MstxCrossRecord record = {
        .addr = reinterpret_cast<uint64_t>(msg),
        .flagId = 0,
        .pipe = pipe_t::PIPE_S,
        .isMore = true,
    };
    __mstx_dfx_report_stub(1, sizeof(MstxCrossRecord), &record);
#endif

    if (!(static_cast<bool>(KfcMsgGetState(msg->head)))) {
        return nullptr;
    }
    msgRcvPos++;
    if (msgRcvPos >= MAX_MSG_COUNT) {
        msgRcvPos = 0;
        msgRcvHead = msgRcvStart;
    } else {
        msgRcvHead++;
    }
    return msg;
}

__aicore__ inline void FreeMessageImpl(__gm__ KfcMsg *msg)
{
    ASCENDC_ASSERT((msg != nullptr),
                   { KERNEL_LOG(KERNEL_ERROR, "msg can not be nullptr"); });
    __asm__ __volatile__("" ::: "memory");
    *(reinterpret_cast<__gm__ uint64_t *>(msg)) = 0;
    Barrier();
    dcci(reinterpret_cast<__gm__ int64_t *>(msg), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    Barrier();
}

__aicore__ inline void RollBackMsgImpl(__gm__ KfcMsg *&msgRcvHead, uint8_t &msgRcvPos)
{
    if (msgRcvPos == 0) {
        msgRcvPos = MAX_MSG_COUNT;
        msgRcvHead = msgRcvHead + MAX_MSG_COUNT -1;
    } else {
        msgRcvPos--;
        msgRcvHead--;
    }
}

}  // namespace AscendC
#endif  // __KERNEL_KFC_COMM_H__
