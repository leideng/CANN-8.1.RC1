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
 * \file moe_distribute_combine.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "moe_distribute_combine.h"
#include "moe_distribute_combine_tiling.h"
#include "moe_distribute_combine_a2.h"
#include "moe_distribute_combine_a2_layered.h"
using namespace AscendC;
using namespace MoeDistributeCombineImpl;
using namespace MoeDistributeCombineA2Impl;
extern "C" __global__ __aicore__ void moe_distribute_combine(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                                             GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount,
                                                             GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale,
                                                             GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR XOut, 
                                                             GM_ADDR workspaceGM, GM_ADDR tilingGM)

{
  REGISTER_TILING_DEFAULT(MoeDistributeCombineA2TilingData);
  REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 2000", MoeDistributeCombineTilingData);
  REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR == 2000) || (TILING_KEY_VAR == 3000)", MoeDistributeCombineA2TilingData);
  TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
  if (TILING_KEY_IS(1100)) { // tp=2
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineTilingData, tilingData, tilingGM);
    MoeDistributeCombine<DTYPE_EXPAND_X, int32_t, true> op;
    op.Init(expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, XOut, workspaceGM, &pipe, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(1000)) { // tp=1
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineTilingData, tilingData, tilingGM);
    MoeDistributeCombine<DTYPE_EXPAND_X, int32_t, false> op;
    op.Init(expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, XOut, workspaceGM, &pipe, &tilingData);
    op.Process();
  }
  if (TILING_KEY_IS(2000)) {
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
    auto tiling = (__gm__ MoeDistributeCombineA2TilingData*)tilingGM;
    __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));
    __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));
    MoeDistributeCombineA2<DTYPE_EXPAND_X, int32_t> op;
    op.Init(expandX, expertIds, expandIdx, epSendCount, scales, XOut, workspaceGM, &pipe, &tilingData,
      mc2InitTiling, mc2CcTiling);
    op.Process();
  }
  if (TILING_KEY_IS(3000)) {
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
    auto tiling = (__gm__ MoeDistributeCombineA2TilingData*)tilingGM;
    __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));
    __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));
    MoeDistributeCombineA2Layered<DTYPE_EXPAND_X, int32_t> op;
    op.Init(expandX, expertIds, expandIdx, epSendCount, expandScales, XOut, workspaceGM, &pipe, &tilingData,
      mc2InitTiling, mc2CcTiling);
    op.Process();
  }
#endif
}