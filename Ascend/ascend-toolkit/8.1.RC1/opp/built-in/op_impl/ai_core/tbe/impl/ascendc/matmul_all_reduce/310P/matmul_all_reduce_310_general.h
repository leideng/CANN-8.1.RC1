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
 * \file matmul_all_reduce_310_general.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_310_GENERAL_H
#define MATMUL_ALL_REDUCE_310_GENERAL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#ifdef __CCE_KT_TEST__
#include "rac_server_stub.h"
#else
#include "rac_server.h"
#endif
#include "../common.h"
#include "mm_allreduce.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <class A_TYPE, class B_TYPE, class BIAS_TYPE, class C_TYPE, bool L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
class MatmulAllReduce310General {
public:
    __aicore__ inline MatmulAllReduce310General() {}
    __aicore__ inline void Init(GM_ADDR workspaceGM, RCSTiling *cfg, Mc2Msg *msg,
                                TCubeTiling *tiling, HcclServer *hcclServer);
    __aicore__ inline void Process(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR dequantGM,
                                   GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR cGM,
                                   GM_ADDR workspaceGM, RCSTiling *cfg, Mc2Msg *msg, TCubeTiling *tiling,
                                   TCubeTiling *tailTiling, L2cacheTilePara *tileL2cacheTiling,
                                   L2cacheTilePara *tailL2cacheTiling, TPipe *tPipe, HcclServer *hcclServer);
private:
    using A_T = typename A_TYPE::T;
    LocalTensor<uint8_t> mmFormatUb_;
};

template <class A_TYPE, class B_TYPE, class BIAS_TYPE, class C_TYPE, bool L2Cache, bool WeightQuant,
    AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulAllReduce310General<A_TYPE, B_TYPE, BIAS_TYPE, C_TYPE, L2Cache, WeightQuant, antiQuantType,
    hasAntiQuantOffset>::Init(GM_ADDR workspaceGM, RCSTiling *cfg, Mc2Msg *msg, TCubeTiling *tiling,
    HcclServer *hcclServer)
{
    TBuf<TPosition::VECCALC> tmpBuf;
    GetTPipePtr()->InitBuffer(tmpBuf, TOTAL_UB_SIZE);
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    __gm__ uint8_t *workspaceMsg = (__gm__ uint8_t *)(context->WorkSpace + msg->notifyOff);
    hcclServer->Init(workspaceMsg, msg->debugMode, tmpBuf);

    workspaceGM += cfg->nd2NzWorkLen;
    workspaceGM += cfg->biasLen;
    GM_ADDR softSyncAddr = workspaceGM;
    hcclServer->InitSoftSync(softSyncAddr, tiling->usedCoreNum, tmpBuf);
    workspaceGM += AC_MAX_AIV * DEFAULT_BLK_NUM;

    int32_t bufferRatio = 2;
    mmFormatUb_ = tmpBuf.Get<uint8_t>(tiling->transLength * bufferRatio);
}

template <class A_TYPE, class B_TYPE, class BIAS_TYPE, class C_TYPE, bool L2Cache, bool WeightQuant,
    AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulAllReduce310General<A_TYPE, B_TYPE, BIAS_TYPE, C_TYPE, L2Cache, WeightQuant, antiQuantType,
    hasAntiQuantOffset>::Process(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR dequantGM, GM_ADDR antiquantScaleGM,
    GM_ADDR antiquantOffsetGM, GM_ADDR cGM, GM_ADDR workspaceGM, RCSTiling *cfg, Mc2Msg *msg,
    TCubeTiling *tiling, TCubeTiling *tailTiling, L2cacheTilePara *tileL2cacheTiling,
    L2cacheTilePara *tailL2cacheTiling, TPipe *tPipe, HcclServer *hcclServer)
{
    MatMulKernel_AllReduce<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, L2Cache, false, false, WeightQuant, antiQuantType,
        hasAntiQuantOffset>(aGM, bGM, cGM, biasGM, dequantGM, *tiling, *cfg, *tileL2cacheTiling, hcclServer,
        cfg->tileCnt, (cfg->tailM ? false : true), false, mmFormatUb_, antiquantScaleGM, antiquantOffsetGM);
    if (cfg->tailM) { // 存在尾块
        aGM = GetTailA(aGM, *tiling, cfg->tileCnt);
        cGM = GetTailC(cGM, *tiling, cfg->tileCnt);
        MatMulKernel_AllReduce<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, L2Cache, false, false,
            WeightQuant, antiQuantType, hasAntiQuantOffset>(aGM, bGM, cGM, biasGM, dequantGM, *tailTiling,
            *cfg, *tailL2cacheTiling, hcclServer, cfg->tailCnt, true, true, mmFormatUb_, antiquantScaleGM,
            antiquantOffsetGM);
    }
}

__aicore__ inline void MatMulEmptyTensorBrcBias(GM_ADDR biasGM, GM_ADDR cGM, MatmulAllReduceTilingData *tilingData,
    TBuf<TPosition::VECCALC> &tmpBuf)
{
    // 搬运biase对齐部分
    int32_t cSizeHalf = (tilingData->param.rankN * tilingData->param.rankM) * sizeof(DTYPE_X1) / sizeof(DTYPE_Y);
    GlobalTensor<DTYPE_Y> cGlobalHalf;
    cGlobalHalf.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(cGM), cSizeHalf);
    LocalTensor<DTYPE_Y> bias = tmpBuf.Get<DTYPE_Y>();
    GlobalTensor<DTYPE_Y> biasGlobal;
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(biasGM));
    uint32_t eleCnt = 32 / sizeof(DTYPE_Y);
    uint32_t alSize = (tilingData->param.rankN / eleCnt) * eleCnt;
    if (alSize > 0) {
        DataCopy(bias, biasGlobal, alSize);
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventID);
        WaitFlag<HardEvent::MTE2_MTE3>(eventID);
        for (uint32_t i = 0; i < tilingData->param.rankM; ++i) {
            uint32_t offsetDst = i * tilingData->param.rankN;
            DataCopy(cGlobalHalf[offsetDst], bias, alSize);
        }
    }
    if (tilingData->param.rankN % eleCnt) {
        // 搬运biase非对齐部分
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);
        DataCopy(bias, biasGlobal[tilingData->param.rankN - eleCnt], eleCnt);
        eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventID);
        WaitFlag<HardEvent::MTE2_MTE3>(eventID);
        for (uint32_t i = 0; i < tilingData->param.rankM; ++i) {
            uint32_t offsetDst = (i + 1) * tilingData->param.rankN - eleCnt;
            DataCopy(cGlobalHalf[offsetDst], bias, eleCnt);
        }
    }
}

__aicore__ inline void MatMulEmptyTensorKernel(GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    MatmulAllReduceTilingData *tilingData, HcclServer *hcclServer)
{
    TBuf<TPosition::VECCALC> tmpBuf;
    GetTPipePtr()->InitBuffer(tmpBuf, TOTAL_UB_SIZE);
    // 初始化hccl
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    __gm__ uint8_t *workspaceMsg = (__gm__ uint8_t *)(context->WorkSpace + (tilingData->msg).notifyOff);
    hcclServer->Init(workspaceMsg, (tilingData->msg).debugMode, tmpBuf);
    if (tilingData->matmulTiling.usedCoreNum > 1) {
        hcclServer->InitSoftSync(workspaceGM, tilingData->matmulTiling.usedCoreNum, tmpBuf);
    }
    // 初始化输出tensor
    int32_t cSize = (tilingData->param.rankN * tilingData->param.rankM) * sizeof(DTYPE_X1) / sizeof(int32_t);
    GlobalTensor<int32_t> cGlobal;
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cGM), cSize);
    if (block_idx == 0) {
        InitOutput<int32_t>(cGlobal, cSize, (int32_t)0);
        if (tilingData->matmulTiling.isBias) {
            MatMulEmptyTensorBrcBias(biasGM, cGM, tilingData, tmpBuf);
        }
    }
    // 通知AICPU计算结束
    if (block_idx == 0) {
        hcclServer->TurnNotifyRun(block_idx, tilingData->matmulTiling.usedCoreNum, 1);
        // 通过一个核轮询并清除数据，防止多核之间写后读依赖
        hcclServer->TurnWait(tilingData->param.tileCnt + tilingData->param.tailCnt);
    }
}

#define INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(templateClass, ...)                                         \
    do {                                                                                             \
        GET_TILING_DATA_MEMBER(MatmulAllReduceTilingData, matmulTiling, tiling, tilingGM);           \
        GET_TILING_DATA_MEMBER(MatmulAllReduceTilingData, tailTiling, tailTiling, tilingGM);          \
        GET_TILING_DATA_MEMBER(MatmulAllReduceTilingData, msg, msg, tilingGM);                       \
        GET_TILING_DATA_MEMBER(MatmulAllReduceTilingData, tailL2cacheTiling, tailL2cacheTiling, tilingGM); \
        GET_TILING_DATA_MEMBER(MatmulAllReduceTilingData, tileL2cacheTiling, tileL2cacheTiling, tilingGM); \
        GET_TILING_DATA_MEMBER(MatmulAllReduceTilingData, param, cfg, tilingGM);                       \
        if (msg.debugMode != static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) {                   \
            templateClass<aType, bType, biasType, cType, __VA_ARGS__> op;                            \
            op.Init(workspaceGM, &cfg, &msg, &tiling, &hcclServer);                               \
            op.Process(aGM, bGM, biasGM, dequantGM, antiquantScaleGM, antiquantOffsetGM,             \
                       cGM, workspaceGM, &cfg, &msg, &tiling, &tailTiling, &tileL2cacheTiling,            \
                       &tailL2cacheTiling, &tPipe, &hcclServer);                         \
        }                                                                                            \
    } while (0)
}
#endif // MATMUL_ALL_REDUCE_310_GENERAL_H
