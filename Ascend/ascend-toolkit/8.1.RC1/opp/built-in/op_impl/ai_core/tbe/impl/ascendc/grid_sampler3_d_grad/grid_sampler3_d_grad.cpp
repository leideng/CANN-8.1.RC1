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
 * \file grid_sampler3_d_grad.cpp
 * \brief
 */

#include "grid_sampler3_d_grad.h"

using namespace GridSampler3DGrad;

extern "C" __global__ __aicore__ void grid_sampler3_d_grad(GM_ADDR grad, GM_ADDR x, GM_ADDR grid, GM_ADDR dx,
                                                           GM_ADDR dgrid, GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);
  GM_ADDR gmTensor[6] = {grad, x, grid, dx, dgrid, workspace};

  if (TILING_KEY_IS(1)) {
    GridSampler3DGradNS<float> op;
    op.Init(&tilingData, gmTensor);
    op.Process();
  }
}
