/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file dynamic_quant_update_scatter_v2.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_v2_db.h"
#include "dynamic_quant_update_scatter_v2.h"

using namespace AscendC;
using namespace DynamicQuantUpdateScatterV2NDOpt;

extern "C" __global__ __aicore__ void dynamic_quant_update_scatter_v2(GM_ADDR x, GM_ADDR indices, GM_ADDR var, GM_ADDR varScale,
                                                                      GM_ADDR varOffset, GM_ADDR varOut, GM_ADDR varScaleOut,
                                                                      GM_ADDR varOffsetOut,GM_ADDR workSpace, GM_ADDR tiling) {
  if (x == nullptr || indices == nullptr || var == nullptr || varScale == nullptr || varOffset == nullptr) {
    return;
  }

  GM_ADDR user1 = GetUserWorkspace(workSpace);
  if (user1 == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);
  if (GetBlockIdx() >= tilingData.coreNum) {
    return;
  }
  TPipe pipe;
  if (TILING_KEY_IS(0) || TILING_KEY_IS(1)) {
    DynamicQuantUpdateScatterV2<DTYPE_X, DTYPE_VAR> op(&pipe);
    op.Init(x, indices, var, varScale, varOffset, workSpace, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(2) || TILING_KEY_IS(3)) {
    DynamicQuantUpdateScatterV2DbOpt<DTYPE_X, DTYPE_VAR> op(&pipe);
    op.Init(x, indices, var, varScale, varOffset, workSpace, &tilingData);
    op.Process();
  } else {
  }
}