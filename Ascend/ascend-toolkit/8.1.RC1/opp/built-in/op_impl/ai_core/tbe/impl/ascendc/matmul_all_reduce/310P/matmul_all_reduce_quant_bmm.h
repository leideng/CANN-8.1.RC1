/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
 * \file matmul_all_reduce_quant_bmm.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_QUANT_BMM_H
#define MATMUL_ALL_REDUCE_QUANT_BMM_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#ifdef __CCE_KT_TEST__
#include "rac_server_stub.h"
#else
#include "rac_server.h"
#endif
#include "../common.h"
#include "mm_allreduce.h"
#include "../../quant_batch_matmul_v3/quant_batch_matmul_v3.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
using namespace DequantBmm;
template <typename aType, typename bType, typename biasType, typename cType, bool aTrans, bool bTrans>
class MatmulAllReduceQuantBmm {
public:
    __aicore__ inline MatmulAllReduceQuantBmm() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR dequantGM, GM_ADDR biasGM, GM_ADDR cGM,
                                GM_ADDR workspaceGM, QuantMatmulAllReduceTilingData* tilingData,
                                TPipe* tPipe, HcclServer* hcclServer);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess(uint32_t tileCnt, QuantBatchMatmulV3TilingData& quant_tiling, uint32_t shift);

private:
    QuantMatmulAllReduceTilingData* tilingData_;
    HcclServer* hcclServer_;
    TPipe* tPipe_;
    GM_ADDR cGM_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR dequantGM_;
    GM_ADDR workspaceGM_;
    TBuf<TPosition::VECCALC> tmpBuf_;
};

template <typename aType, typename bType, typename biasType, typename cType, bool aTrans, bool bTrans>
__aicore__ inline void MatmulAllReduceQuantBmm<aType, bType, biasType, cType, aTrans, bTrans>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR dequantGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    QuantMatmulAllReduceTilingData* tilingData, TPipe* tPipe, HcclServer* hcclServer) {
    GetTPipePtr()->InitBuffer(tmpBuf_, TOTAL_UB_SIZE);
    auto&& cfg = tilingData->param;

    __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());
    __gm__ uint8_t* workspaceMsg = (__gm__ uint8_t*)(context->WorkSpace + (tilingData->msg).notifyOff);
    hcclServer->Init(workspaceMsg, (tilingData->msg).debugMode, tmpBuf_);

    workspaceGM += cfg.nd2NzWorkLen;
    workspaceGM += cfg.biasLen;
    GM_ADDR softSyncAddr = workspaceGM;
    hcclServer->InitSoftSync(softSyncAddr, GetBlockNum(), tmpBuf_);
    workspaceGM += AC_MAX_AIV * DEFAULT_BLK_NUM;

    tilingData_ = tilingData;
    hcclServer_ = hcclServer;
    tPipe_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    dequantGM_ = dequantGM;
    workspaceGM_ = workspaceGM;
}

template <typename aType, typename bType, typename biasType, typename cType, bool aTrans, bool bTrans>
__aicore__ inline void MatmulAllReduceQuantBmm<aType, bType, biasType, cType, aTrans, bTrans>::Process() {
    auto&& cfg = tilingData_->param;
    if (g_coreType == AIV) {
        return;
    }

    InnerProcess(cfg.tileCnt, tilingData_->tilematmulTiling, 0);
    if (cfg.tailM != 0U) {
        InnerProcess(cfg.tailCnt, tilingData_->tailmatmulTiling, cfg.tileCnt);
    }

    if (GetBlockIdx() == 0 && (g_coreType == AIC || g_coreType == MIX)) {
        hcclServer_->TurnWait(cfg.tileCnt + cfg.tailCnt);
    }
}

template <typename aType, typename bType, typename biasType, typename cType, bool aTrans, bool bTrans>
__aicore__ inline void MatmulAllReduceQuantBmm<aType, bType, biasType, cType, aTrans, bTrans>::InnerProcess(
    uint32_t tileCnt, QuantBatchMatmulV3TilingData& quant_tiling, uint32_t shift) {
    const uint64_t aOffset = CalcShapeOffset(sizeof(aType), quant_tiling.matmulTiling.M, quant_tiling.matmulTiling.Ka);
    const uint64_t cOffset = CalcShapeOffset(sizeof(cType), quant_tiling.matmulTiling.M, quant_tiling.matmulTiling.N);
    BmmDequant<aType, bType, FORMAT_X1, FORMAT_X2, biasType, uint64_t, cType, aTrans, bTrans> op;
    for (uint32_t i = 0; i < tileCnt; i++) {
        if (i == 0) {
            tPipe_->Reset();
            op.InitTbuf(tmpBuf_);
            op.Init(aGM_, bGM_, biasGM_, dequantGM_, cGM_, workspaceGM_, &quant_tiling, tPipe_);
        } else {
            op.UpdateGlobalAddr(aGM_, bGM_, biasGM_, dequantGM_, cGM_, workspaceGM_);
        }

        op.Process();
        hcclServer_->TurnNotifyRun(block_idx, GetBlockNum(), i + 1 + shift);
        aGM_ += aOffset;
        cGM_ += cOffset;
    }
}

#define INVOKE_QUANT_BMM_OP_IMPL(templateClass, ...)                                                   \
    do {                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(QuantMatmulAllReduceTilingData, tilingData, tilingGM);             \
        templateClass<DTYPE_X1, DTYPE_X2, int32_t, DTYPE_Y, __VA_ARGS__> op;                           \
        op.Init(aGM, bGM, dequantGM, biasGM, cGM, workspaceGM, &tilingData,                            \
                &tPipe, &hcclServer);                                                                  \
        op.Process();                                                                                  \
    } while (0)
}  // namespace MatmulAllReduceImpl
#endif  // MATMUL_ALL_REDUCE_QUANT_BMM_H