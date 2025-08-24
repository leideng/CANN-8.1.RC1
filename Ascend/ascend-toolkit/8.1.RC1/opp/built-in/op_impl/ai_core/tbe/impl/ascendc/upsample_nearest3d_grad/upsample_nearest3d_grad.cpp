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
 * \file upsample_nearest3d_grad.cpp
 * \brief
 */

#include "upsample_nearest3d_grad.h"

using namespace UpsampleNearest3dGrad;

extern "C" __global__ __aicore__ void upsample_nearest3d_grad(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  const UpsampleNearest3dGradTilingData* __restrict tiling_data = &tilingData;
  const TCubeTiling* __restrict matmulTilingWTiling = &(tiling_data->matmulTilingW);
  const TCubeTiling* __restrict matmulTilingHTiling = &(tiling_data->matmulTilingH);
  const TCubeTiling* __restrict matmulTilingDTiling = &(tiling_data->matmulTilingD);

  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

#define INIT_AND_PROCESS                                                                                              \
  REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling, \
                    op.matmulD, matmulTilingDTiling);                                                                 \
  op.Init(x, y, false, userWS, &tilingData);                                                                          \
  op.Process()

  if (TILING_KEY_IS(1)) {
    if (tiling_data->dataType == 1) {
      UpsampleNearest3dGradND<half> op;
      INIT_AND_PROCESS;
    } else if (tiling_data->dataType == 2) {
      UpsampleNearest3dGradND<float> op;
      INIT_AND_PROCESS;
    } else if (tiling_data->dataType == 3) {
      UpsampleNearest3dGradND<bfloat16_t> op;
      INIT_AND_PROCESS;
    }
  }
}
