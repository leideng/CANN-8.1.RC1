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
 * \file mm_allreduce.h
 * \brief
 */
#ifndef MC2_MM_ALLREDUCE_H
#define MC2_MM_ALLREDUCE_H

#ifdef __CCE_KT_TEST__
#include "rac_server_stub.h"
#else
#include "rac_server.h"
#endif
#include "matmul_compute_weight_quant.h"

namespace AscendC {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool L2Cache = false, bool IS_QUANT = false,
    bool IS_PER_TENSOR = false, bool WeightQuant = false, AntiQuantType antiQuantType = AntiQuantType::NONE,
    bool hasAntiQuantOffset = false>
__aicore__ inline void MatMulKernel_AllReduce_WeightQuant(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
    GM_ADDR dequantGM, TCubeTiling& tiling, RCSTiling& cfg, L2cacheTilePara& tileL2cacheTiling, HcclServer* hcclServer,
    uint32_t tileCnt, bool isTail, const LocalTensor<uint8_t>& mmFormatUb, GM_ADDR antiquantScale,
    GM_ADDR antiquantOffset)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

    auto aOffset = tiling.M *  tiling.Ka * sizeof(A_T);
    auto cOffset = tiling.M *  tiling.N  * sizeof(C_T);
    // 处理带C‘场景
    uint32_t indexC = 0;
    uint8_t enAtomicC = 0;
    if (cfg.isAdd) {
        enAtomicC = 1;
    }
    MatmulComputeWeightQuant<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, L2Cache, WeightQuant, antiQuantType,
        hasAntiQuantOffset> mm;
    // AllReduce需要提前计算一次C矩阵的Offset地址
    mm.Init(tiling, cfg, tileL2cacheTiling, mmFormatUb);
    mm.InitGlobalBTensor(bGM, biasGM, antiquantScale, antiquantOffset);
    auto aAddr = aGM;
    auto cAddr = cGM;

    // 处理尾块时CurTurn需要偏移tileCnt个单位
    auto shift = isTail ? cfg.tileCnt : 0;
    for (uint32_t i = 0; i < tileCnt; i++) {
        mm.InitGlobalATensor(aAddr, aOffset, cAddr, cOffset);
        mm.Compute(indexC, enAtomicC);
        hcclServer->TurnNotifyRun(block_idx, tiling.usedCoreNum, i+1+shift);
        aAddr += aOffset;
        cAddr += cOffset;
    }
    mm.End();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool L2Cache = false, bool IS_QUANT = false,
    bool IS_PER_TENSOR = false, bool WeightQuant = false, AntiQuantType antiQuantType = AntiQuantType::NONE,
    bool hasAntiQuantOffset = false>
__aicore__ inline void MatMulKernel_AllReduce(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR dequantGM,
                                              TCubeTiling& tiling, RCSTiling& cfg,
                                              L2cacheTilePara& tileL2cacheTiling, HcclServer* hcclServer,
                                              uint32_t tileCnt, bool isLast, bool isTail,
                                              const LocalTensor<uint8_t>& mmFormatUb, GM_ADDR antiquantScale,
                                              GM_ADDR antiquantOffset)
{
    if (g_coreType == AIV) {
        return;
    }
    if (block_idx >= tiling.usedCoreNum) {
        for (uint32_t i = 0; i < tileCnt; i++) {
#if __CCE_AICORE__ == 220
            hcclServer->TurnNotifyRun(block_idx);
#endif
        }
        return;
    }
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

    auto aOffset = tiling.M *  tiling.Ka * sizeof(A_T);
    auto cOffset = tiling.M *  tiling.N  * sizeof(C_T);
    // 处理带C‘场景
    uint32_t indexC = 0;
    uint8_t enAtomicC = 0;
    if (cfg.isAdd) {
        enAtomicC = 1;
    }
    // 归一化Matmul计算类，负责MC2的Matmul计算
    if constexpr (WeightQuant) {
        MatMulKernel_AllReduce_WeightQuant<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, L2Cache, false, false, WeightQuant,
            antiQuantType, hasAntiQuantOffset>(aGM, bGM, cGM, biasGM, dequantGM, tiling, cfg, tileL2cacheTiling,
            hcclServer, tileCnt, isTail, mmFormatUb, antiquantScale, antiquantOffset);

    } else {
        MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, L2Cache, WeightQuant, antiQuantType, hasAntiQuantOffset> mm;
        // AllReduce需要提前计算一次C矩阵的Offset地址
        mm.Init(tiling, cfg, tileL2cacheTiling, mmFormatUb);
        mm.InitGlobalBTensor(bGM, biasGM, antiquantScale, antiquantOffset);
        auto aAddr = aGM;
        auto cAddr = cGM;
 
        // 处理尾块时CurTurn需要偏移tileCnt个单位
        auto shift = isTail ? cfg.tileCnt : 0;
        for (uint32_t i = 0; i < tileCnt; i++) {
            mm.InitGlobalATensor(aAddr, aOffset, cAddr, cOffset);
            mm.Compute(indexC, enAtomicC);
            hcclServer->TurnNotifyRun(block_idx, tiling.usedCoreNum, i+1+shift);
            aAddr += aOffset;
            cAddr += cOffset;
        }
        mm.End();
    }
    // 通过一个核轮询并清除数据，防止多核之间写后读依赖
    if (isLast && GetBlockIdx() == 0  && (g_coreType == AIC || g_coreType == MIX)) {
        hcclServer->TurnWait(cfg.tileCnt + cfg.tailCnt);
    }
}
}
#endif // MC2_MM_ALLREDUCE_H
