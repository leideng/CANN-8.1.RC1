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
 * \file dynamic_quant_v2.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "../dynamic_quant/dynamic_quant.h"
#include "../dynamic_quant/dynamic_quant_db.h"
#include "../dynamic_quant/dynamic_quant_large_shape_opt.h"
#include "../dynamic_quant/dynamic_quant_moe.h"
#include "../dynamic_quant/dynamic_quant_moe_large_shape.h"

using namespace AscendC;
using namespace DynamicQuantNDOpt;

extern "C" __global__ __aicore__ void dynamic_quant_v2(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index, GM_ADDR y,
                                                       GM_ADDR scale, GM_ADDR offset, GM_ADDR workSpace,
                                                       GM_ADDR tiling) {
  if (x == nullptr || y == nullptr || scale == nullptr) {
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
    DynamicQuant<DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, smooth_scales, y, scale, offset, workSpace, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(2) || TILING_KEY_IS(3)) {
    DynamicQuantDbOpt<DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, smooth_scales, y, scale, offset, workSpace, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(6)) {
    DynamicQuantLargeShapeOpt<DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, smooth_scales, y, scale, offset, workSpace, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(7)) {
    DynamicQuantMoe<DTYPE_X, int32_t, DTYPE_Y> op(&pipe);
    op.Init(x, smooth_scales, group_index, y, scale, offset, workSpace, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(8)) {
    DynamicQuantMoeLargeShape<DTYPE_X, DTYPE_X, int32_t, DTYPE_Y> op(&pipe);
    op.Init(x, smooth_scales, group_index, y, scale, offset, workSpace, &tilingData);
    op.Process();
  } else {
  }
}