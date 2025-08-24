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
 * \file matmul_all_reduce_weight_quant_310.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_WEIGHT_QUANT_310_H
#define MATMUL_ALL_REDUCE_WEIGHT_QUANT_310_H
 
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#ifdef __CCE_KT_TEST__
#include "rac_server_stub.h"
#else
#include "rac_server.h"
#endif
#include "../common.h"
#include "mm_allreduce.h"
#include "../../weight_quant_batch_matmul_v2/weight_quant_batch_matmul_v2_weight_nz_performance.h"
 
namespace MatmulAllReduceImpl {
using namespace AscendC;
using namespace WeightQuantBatchMatmulV2;
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          QuantType antiQuantType, bool hasAntiQuantOffset>
class MatmulAllReduceWeightQuant310 {
public:
    __aicore__ inline MatmulAllReduceWeightQuant310() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM,
                                GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                WeightQuantMatmulAllReduceNzTilingData *tilingData, TPipe *tPipe,
                                HcclServer *hcclServer);
    __aicore__ inline void Process();
 
private:
    __aicore__ inline void InnerProcess(uint32_t tileCnt, WeightQuantBatchMatmulV2NzTilingData *mmTiling,
        uint32_t shift, int32_t coreNum);
    WeightQuantMatmulAllReduceNzTilingData *tilingData_;
    HcclServer *hcclServer_;
    TPipe *tPipe_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR antiquantScaleGM_;
    GM_ADDR antiquantOffsetGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR cGM_;
    bool notifyFlag_{false};
    int32_t coreNum_{0};
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulAllReduceWeightQuant310<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
hasAntiQuantOffset>::InnerProcess(
    uint32_t tileCnt, WeightQuantBatchMatmulV2NzTilingData *mmTiling, uint32_t shift, int32_t coreNum) {
    const uint64_t aOffset = CalcShapeOffset(sizeof(xType), mmTiling->mSize, mmTiling->kSize);
    const uint64_t cOffset = CalcShapeOffset(sizeof(yType), mmTiling->mSize, mmTiling->nSize);
    if (GetBlockIdx() < coreNum) {
        WeightQuantBatchMatmulV2WeightNzPerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans,
            antiQuantType, hasAntiQuantOffset> op;
        for (uint32_t i = 1U; i <= tileCnt; ++i) {
            if (i == 1U) {
                tPipe_->Reset();
                op.Init(aGM_, bGM_, antiquantScaleGM_, antiquantOffsetGM_, nullptr, nullptr, biasGM_, cGM_,
                        workspaceGM_, mmTiling, tPipe_);
            } else {
                op.UpdateGlobalAddr(aGM_, bGM_, antiquantScaleGM_, antiquantOffsetGM_, nullptr, nullptr, biasGM_,
                                    cGM_, workspaceGM_);
            }
            op.Process();
            hcclServer_->TurnNotifyRun(block_idx, coreNum, i + shift);
            aGM_ += aOffset;
            cGM_ += cOffset;
        }
    } else {
        for (uint32_t i = 1U; i <= tileCnt; ++i) {
            aGM_ += aOffset;
            cGM_ += cOffset;
        }
    }
}
 
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulAllReduceWeightQuant310<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
hasAntiQuantOffset>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR biasGM, GM_ADDR cGM,
    GM_ADDR workspaceGM, WeightQuantMatmulAllReduceNzTilingData *tilingData, TPipe *tPipe,
    HcclServer *hcclServer) {
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    __gm__ uint8_t *workspaceMsg = (__gm__ uint8_t *)(context->WorkSpace + tilingData->msg.notifyOff);
    TBuf<TPosition::VECCALC> tmpBuf;
    tPipe->InitBuffer(tmpBuf, 256);
    auto&& cfg = tilingData->param;
    auto&& tiling = tilingData->tilematmulTiling.matmulTiling;
    hcclServer->Init(workspaceMsg, (tilingData->msg).debugMode, tmpBuf);
    workspaceGM += cfg.nd2NzWorkLen;
    workspaceGM += cfg.biasLen;
    GM_ADDR softSyncAddr = workspaceGM;
    int32_t tileNum = tilingData->tilematmulTiling.cubeBlockDimN * tilingData->tilematmulTiling.cubeBlockDimM;
    int32_t tailNum = tilingData->tailmatmulTiling.cubeBlockDimN * tilingData->tailmatmulTiling.cubeBlockDimM;
    coreNum_ = tileNum > tailNum ? tileNum : tailNum;
    hcclServer->InitSoftSync(softSyncAddr, coreNum_, tmpBuf);
    workspaceGM += AC_MAX_AIV * DEFAULT_BLK_NUM;
 
    if (tilingData->msg.useBufferType == MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_WINDOW_IN
        && context->config.determinism != 1) {
        cGM_ = (__gm__ uint8_t *)(context->windowsIn[context->rankId]);
    } else {
        cGM_ = cGM;
    }
    tilingData_ = tilingData;
    hcclServer_ = hcclServer;
    tPipe_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    biasGM_ = biasGM;
    antiquantScaleGM_ = antiquantScaleGM;
    antiquantOffsetGM_ = antiquantOffsetGM;
    workspaceGM_ = workspaceGM;
    if (GetBlockIdx() == 0 && (g_coreType == AIC || g_coreType == MIX)) {
        notifyFlag_ = true;
    }
}
 
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulAllReduceWeightQuant310<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
hasAntiQuantOffset>::Process() {
    auto &&mc2Tiling = tilingData_->param;
    int32_t tileNum = tilingData_->tilematmulTiling.cubeBlockDimN * tilingData_->tilematmulTiling.cubeBlockDimM;
    int32_t tailNum = tilingData_->tailmatmulTiling.cubeBlockDimN * tilingData_->tailmatmulTiling.cubeBlockDimM;
    InnerProcess(mc2Tiling.tileCnt, &tilingData_->tilematmulTiling, 0U, tileNum);
    if (mc2Tiling.tailM != 0U) {
        InnerProcess(mc2Tiling.tailCnt, &tilingData_->tailmatmulTiling, mc2Tiling.tileCnt, tailNum);
    }
    if (notifyFlag_) {
        hcclServer_->TurnWait(mc2Tiling.tileCnt + mc2Tiling.tailCnt);
    }
}
 
__aicore__ inline void MatMulEmptyTensorBrcBias(GM_ADDR biasGM, GM_ADDR cGM,
    WeightQuantMatmulAllReduceNzTilingData *tilingData, TBuf<TPosition::VECCALC> &tmpBuf)
{
    // 搬运biase对齐部分
    int32_t cSizeHalf = (tilingData->param.rankN * tilingData->param.rankM) * sizeof(DTYPE_X1) / sizeof(DTYPE_Y);
    GlobalTensor<DTYPE_Y> cGlobalHalf;
    cGlobalHalf.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(cGM), cSizeHalf);
    LocalTensor<DTYPE_Y> bias = tmpBuf.Get<DTYPE_Y>();
    GlobalTensor<DTYPE_Y> biasGlobal;
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(biasGM));
    TBuffAddr buffAddr;
    buffAddr.logicPos = (uint8_t)QuePosition::VECCALC;
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
 
__aicore__ inline void WeightQuantEmptyTensorKernel(GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    WeightQuantMatmulAllReduceNzTilingData *tilingData, HcclServer *hcclServer)
{
    TBuf<TPosition::VECCALC> tmpBuf;
    GetTPipePtr()->InitBuffer(tmpBuf, TOTAL_UB_SIZE);
    // 初始化hccl
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    __gm__ uint8_t *workspaceMsg = (__gm__ uint8_t *)(context->WorkSpace + (tilingData->msg).notifyOff);
    hcclServer->Init(workspaceMsg, (tilingData->msg).debugMode, tmpBuf);
    if (tilingData->tilematmulTiling.matmulTiling.usedCoreNum > 1) {
        hcclServer->InitSoftSync(workspaceGM, tilingData->tilematmulTiling.matmulTiling.usedCoreNum, tmpBuf);
    }
    // 初始化输出tensor
    int32_t cSize = (tilingData->param.rankN * tilingData->param.rankM) * sizeof(DTYPE_X1) / sizeof(int32_t);
    GlobalTensor<int32_t> cGlobal;
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cGM), cSize);
    if (block_idx == 0) {
        InitOutput<int32_t>(cGlobal, cSize, (int32_t)0);
        if (tilingData->tilematmulTiling.matmulTiling.isBias) {
            MatMulEmptyTensorBrcBias(biasGM, cGM, tilingData, tmpBuf);
        }
    }
    // 通知AICPU计算结束
    if (block_idx == 0) {
        hcclServer->TurnNotifyRun(block_idx, tilingData->tilematmulTiling.matmulTiling.usedCoreNum, 1);
        // 通过一个核轮询并清除数据，防止多核之间写后读依赖
        hcclServer->TurnWait(tilingData->param.tileCnt + tilingData->param.tailCnt);
    }
}

#define INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(templateClass, ...)                                        \
    do {                                                                                               \
        GET_TILING_DATA_MEMBER(WeightQuantMatmulAllReduceNzTilingData, msg, msg, tilingGM);            \
        if (msg.debugMode != static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) {                  \
            GET_TILING_DATA_WITH_STRUCT(WeightQuantMatmulAllReduceNzTilingData, tilingData, tilingGM); \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_Y, __VA_ARGS__> op;                       \
            op.Init(aGM, bGM, antiquantScaleGM, antiquantOffsetGM, biasGM, cGM, userWS,                \
                    &tilingData, &tPipe, &hcclServer);                                                 \
            op.Process();                                                                              \
        }                                                                                              \
    } while (0)
}  // namespace MatmulAllReduceImpl
#endif  // MATMUL_ALL_REDUCE_WEIGHT_QUANT_310_H