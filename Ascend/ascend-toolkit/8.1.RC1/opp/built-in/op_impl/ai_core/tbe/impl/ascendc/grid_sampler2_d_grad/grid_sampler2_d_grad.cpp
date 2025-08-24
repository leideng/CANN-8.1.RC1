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
 * \file grid_sampler2_d_grad.cpp
 * \brief
 */

#include "grid_sampler2_d_grad.h"
#include "grid_sampler2_d_grad_fp16.h"
#include "grid_sampler2_d_grad_cast.h"


__aicore__ inline void InitWorkspace(const GridSampler2DGradTilingData &tiling, GM_ADDR workSpace)
{
  uint32_t blockIdx = GetBlockIdx();
  uint32_t computePNum = 0;
  uint32_t castOffset = 0;
  float initParam = 0.0f;
  uint32_t tailPNumCast = tiling.tailPNumCast;
  uint32_t pNumPerCoreCast = tiling.pNumPerCoreCast;
  GlobalTensor<float> indexCountGm;
  if (blockIdx < tailPNumCast) {
    computePNum = pNumPerCoreCast + 1;
    castOffset = blockIdx * computePNum;
  } else {
    computePNum = pNumPerCoreCast;
    castOffset = blockIdx * computePNum + tailPNumCast;
  }
  indexCountGm.SetGlobalBuffer((__gm__ float*)workSpace + castOffset);
  InitOutput(indexCountGm, computePNum, initParam);
}


// kernel function
extern "C" __global__ __aicore__ void grid_sampler2_d_grad(GM_ADDR grad, GM_ADDR x, GM_ADDR grid, GM_ADDR dx,
                                                           GM_ADDR dgrid, GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
    return;
  }
  TPipe pipe;
  GET_TILING_DATA(tilingData, tiling);
  GM_ADDR gmTensor[6] = {grad, x, grid, dx, dgrid, workspace};
  if (TILING_KEY_IS(1)) {
    GridSampler2DGrad<float, GridSampler2DGradTilingData> op;
    op.Init(tilingData, gmTensor);
    op.InitBuffer(&pipe);
    op.InitBilinearLocalTensor();
    op.Process();
  }
  if (TILING_KEY_IS(2)) {
    GridSampler2DGrad<float, GridSampler2DGradTilingData> op;
    op.Init(tilingData, gmTensor);
    op.InitBuffer(&pipe);
    op.InitNearestLocalTensor();
    op.Process();
  }
  if (TILING_KEY_IS(3)) {
    InitWorkspace(tilingData, workspace);
    SyncAll();
    GridSampler2DGradFP16<float, half, GridSampler2DGradTilingData> op;
    op.Init(tilingData, gmTensor);
    op.InitBuffer(&pipe);
    op.InitBilinearLocalTensor();
    op.Process();
    SyncAll();
    pipe.Destroy();
    TPipe tpipe;
    GridSampler2DGradCast<half, GridSampler2DGradTilingData> castOp;
    castOp.Init(tilingData, gmTensor, &tpipe);
    castOp.Process();
  }
  if (TILING_KEY_IS(4)) {
    InitWorkspace(tilingData, workspace);
    SyncAll();
    GridSampler2DGradFP16<float, half, GridSampler2DGradTilingData> op;
    op.Init(tilingData, gmTensor);
    op.InitBuffer(&pipe);
    op.InitNearestLocalTensor();
    op.Process();
    SyncAll();
    pipe.Destroy();
    TPipe tpipe;
    GridSampler2DGradCast<half, GridSampler2DGradTilingData> castOp;
    castOp.Init(tilingData, gmTensor, &tpipe);
    pipe_barrier(PIPE_ALL);
    castOp.Process();
  }
  if (TILING_KEY_IS(5)) {
    InitWorkspace(tilingData, workspace);
    SyncAll();
    GridSampler2DGradFP16<float, bfloat16_t, GridSampler2DGradTilingData> op;
    op.Init(tilingData, gmTensor);
    op.InitBuffer(&pipe);
    op.InitBilinearLocalTensor();
    op.Process();
    SyncAll();
    pipe.Destroy();
    TPipe tpipe;
    GridSampler2DGradCast<bfloat16_t, GridSampler2DGradTilingData> castOp;
    castOp.Init(tilingData, gmTensor, &tpipe);
    castOp.Process();
  }
  if (TILING_KEY_IS(6)) {
    InitWorkspace(tilingData, workspace);
    SyncAll();
    GridSampler2DGradFP16<float, bfloat16_t, GridSampler2DGradTilingData> op;
    op.Init(tilingData, gmTensor);
    op.InitBuffer(&pipe);
    op.InitNearestLocalTensor();
    op.Process();
    SyncAll();
    pipe.Destroy();
    TPipe tpipe;
    GridSampler2DGradCast<bfloat16_t, GridSampler2DGradTilingData> castOp;
    castOp.Init(tilingData, gmTensor, &tpipe);
    castOp.Process();
  }
}