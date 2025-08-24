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
 * \file adaptive_max_pool3d.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "adaptive_max_pool3d_small_pool.h"
#include "adaptive_max_pool3d_big_pool.h"

extern "C" __global__ __aicore__ void adaptive_max_pool3d(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace,
                                                          GM_ADDR tiling) {
  __gm__ uint8_t* user = GetUserWorkspace(workspace);

  TPipe pipeBase;
  if (TILING_KEY_IS(320000UL)) {
    GET_TILING_DATA_WITH_STRUCT(AdaptiveMaxPool3dSmallPoolTilingData, tilingDataIn, tiling);
    const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tilingData = &tilingDataIn;
    AdaptiveMaxPool3dSmallPool<float> op;
    op.Init(x, y, indices, user, &pipeBase, tilingData);
    op.Process();
    return;
  } else if (TILING_KEY_IS(321000UL)) {
    GET_TILING_DATA_WITH_STRUCT(AdaptiveMaxPool3dSmallPoolTilingData, tilingDataIn, tiling);
    const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tilingData = &tilingDataIn;
    AdaptiveMaxPool3dSmallPool<half> op;
    op.Init(x, y, indices, user, &pipeBase, tilingData);
    op.Process();
    return;
  } else if (TILING_KEY_IS(322000UL)) {
    GET_TILING_DATA_WITH_STRUCT(AdaptiveMaxPool3dSmallPoolTilingData, tilingDataIn, tiling);
    const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tilingData = &tilingDataIn;
    AdaptiveMaxPool3dSmallPool<bfloat16_t> op;
    op.Init(x, y, indices, user, &pipeBase, tilingData);
    op.Process();
    return;
  } else if (TILING_KEY_IS(310000UL)) {
    GET_TILING_DATA_WITH_STRUCT(AdaptiveMaxPool3dBigPoolTilingData, tilingDataIn, tiling);
    const AdaptiveMaxPool3dBigPoolTilingData *__restrict tilingData = &tilingDataIn;
    AdaptiveMaxPool3dBigPool<float, float> op;
    op.Init(x, y, indices, GetUserWorkspace(workspace), &pipeBase, tilingData, 0);
    op.Process();
  } else if (TILING_KEY_IS(311000UL)) {
    GET_TILING_DATA_WITH_STRUCT(AdaptiveMaxPool3dBigPoolTilingData, tilingDataIn, tiling);
    const AdaptiveMaxPool3dBigPoolTilingData *__restrict tilingData = &tilingDataIn;
    AdaptiveMaxPool3dBigPool<half, half> op;
    op.Init(x, y, indices, GetUserWorkspace(workspace), &pipeBase, tilingData, 1);
    op.Process();
  } else if (TILING_KEY_IS(312000UL)) {
    GET_TILING_DATA_WITH_STRUCT(AdaptiveMaxPool3dBigPoolTilingData, tilingDataIn, tiling);
    const AdaptiveMaxPool3dBigPoolTilingData *__restrict tilingData = &tilingDataIn;
    AdaptiveMaxPool3dBigPool<bfloat16_t, float> op;
    op.Init(x, y, indices, GetUserWorkspace(workspace), &pipeBase, tilingData, 2);
    op.Process();
  }

  return;
}