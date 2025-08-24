/* *
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

/* !
 * \file matmul_reduce_scatter_base.h
 * \brief
 */
#ifndef MATMUL_REDUCE_SCATTER_BASE_H
#define MATMUL_REDUCE_SCATTER_BASE_H

#include "lib/matmul_intf.h"
#include "mc2_nd_to_nz.h"
#include "reduce_scatter_nd_to_nz.h"
#include "mc2_matmul_compute.h"

namespace AscendC {
constexpr uint8_t MC2_DEBUG_ONLY_CUBE = 1;  // 只计算不通信
constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4; // 只通信不计算

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
class MatmulReduceScatterBase {
public:
    __aicore__ inline MatmulReduceScatterBase() {}
    __aicore__ inline void InitBase(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
        GM_ADDR contextGM, MatmulReduceScatterTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Nd2NzBiasCast();

protected:
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR cGM_;
    GM_ADDR gmToFloat_;
    GM_ADDR workspaceGM_;
    MatmulReduceScatterTilingData *tilingData_;
    TPipe *tPipe_;
    uint32_t rankId_{ 0 };
    uint32_t rankDim_{ 8 };
    bool debugOnlyCalc_{ false };
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void MatmulReduceScatterBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::InitBase(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR contextGM,
    MatmulReduceScatterTilingData *tilingData, TPipe *tPipe)
{
    aGM_ = aGM;
    bGM_ = bGM;
    biasGM_ = biasGM;
    cGM_ = cGM;
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    tPipe_ = tPipe;

    gmToFloat_ = workspaceGM_; // step1, C矩阵float
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void MatmulReduceScatterBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::Nd2NzBiasCast()
{
    TBuf<TPosition::VECCALC> totalUbBuf;
    tPipe_->InitBuffer(totalUbBuf, TOTAL_UB_SIZE);

    // worspace 空间划分:
    GM_ADDR gmNd2NzAddr = workspaceGM_ + (uint64_t)tilingData_->param.cToFloatLen; // step2, ND2NZ 后的地址
    GM_ADDR biasToFloat = gmNd2NzAddr + (uint64_t)tilingData_->param.nd2NzWorkLen; // step3, bias 后的地址
    if constexpr (Bias2Float) {
        CastBFtoFloat(biasToFloat, biasGM_, tilingData_->param.rankN, totalUbBuf);
        biasGM_ = biasToFloat;
    }
    if constexpr (BNd2Nz) {
        MatrixBtoNZMc2<typename B_TYPE::T>(gmNd2NzAddr, bGM_, tilingData_, tilingData_->param.isTransposeB, totalUbBuf);
        bGM_ = gmNd2NzAddr;
    }
}
}
#endif