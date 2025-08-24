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
 * \file constant_tiling.h
 * \brief
 */
#ifndef LIB_MATMUL_CONSTANT_TILING_H
#define LIB_MATMUL_CONSTANT_TILING_H

#include "../../impl/matmul/tiling/matmul_constant_tiling_impl.h"

namespace AscendC {
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ constexpr MatmulApiStaticTiling GetMatmulApiTiling(const MatmulConfig &mmCFG, int32_t l1Size = Impl::L1_SIZE)
{
    MatmulApiStaticTiling tiling;
    tiling.cfg = mmCFG;
    if ((mmCFG.singleCoreM == 0) || (mmCFG.singleCoreN == 0) || (mmCFG.singleCoreK == 0)) {
        if (mmCFG.basicM != 0 && mmCFG.basicN != 0 && mmCFG.basicK != 0) {
            tiling.baseM = mmCFG.basicM;
            tiling.baseN = mmCFG.basicN;
            tiling.baseK = mmCFG.basicK;
            tiling.dbL0A = GetL0ADb(mmCFG, TOTAL_L0A_SIZE);
            tiling.dbL0B = GetL0BDb(mmCFG, TOTAL_L0B_SIZE);
            tiling.isBias = mmCFG.enableSetBias;
        }
        return tiling;
    }
    L1Status l1Factor = GetL1Factor<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG, l1Size);
    // when enable constant tiling, user need update orgShape
    tiling.M = mmCFG.singleCoreM;
    tiling.N = mmCFG.singleCoreN;
    tiling.Ka = mmCFG.singleCoreK;
    tiling.Kb = mmCFG.singleCoreK;
    tiling.singleCoreM = mmCFG.singleCoreM;
    tiling.singleCoreN = mmCFG.singleCoreN;
    tiling.singleCoreK = mmCFG.singleCoreK;
    tiling.baseM = mmCFG.basicM;
    tiling.baseN = mmCFG.basicN;
    tiling.baseK = mmCFG.basicK;
    tiling.stepM = l1Factor.mAL1;
    tiling.stepN = l1Factor.nBL1;
    int32_t reduceC0Size = GetReduceC0Size<typename A_TYPE::T>();
    int32_t kL0 = GetKL0<A_TYPE>(mmCFG);
    tiling.stepKa = CeilNoLog<int32_t>(l1Factor.kAL1, kL0);
    tiling.stepKb = CeilNoLog<int32_t>(l1Factor.kBL1, kL0);
    tiling.depthA1 = CeilNoLog<int32_t>(l1Factor.kAL1, kL0) * l1Factor.mAL1 * l1Factor.dbAL1;
    tiling.depthB1 = CeilNoLog<int32_t>(l1Factor.kBL1, kL0) * l1Factor.nBL1 * l1Factor.dbBL1;
    tiling.iterateOrder = GetIterateOrder<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Factor, mmCFG);
    tiling.isBias = mmCFG.enableSetBias;
    tiling.dbL0A = GetL0ADb(mmCFG, TOTAL_L0A_SIZE);
    tiling.dbL0B = GetL0BDb(mmCFG, TOTAL_L0B_SIZE);
    // keep the same with runtime tiling, fix l0c db
    tiling.dbL0C = 1;
    tiling.transLength = GetTransLength<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG, l1Factor);
    tiling.shareMode = 0;
    tiling.shareL1Size = GetL1UsedSize<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG, l1Factor,
        tiling.depthA1, tiling.depthB1);
    tiling.shareL0CSize = mmCFG.basicM * mmCFG.basicN * GetBitSize<float>() / ONE_BYTE_BIT_SIZE;
    // tiling constant not support v200
    tiling.shareUbSize = 0;
    return tiling;
}

} // namespace matmul
#endif // LIB_MATMUL_CONSTANT_TILING_H
