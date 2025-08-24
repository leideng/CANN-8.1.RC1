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
 * \file upsample_bicubic2d_aa_grad.cpp
 * \brief
 */

#include "upsample_bicubic2d_aa_grad.h"

using namespace UpSampleBicubic2dAAGrad;

extern "C" __global__ __aicore__ void upsample_bicubic2d_aa_grad(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                             GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  const UpsampleBicubicAAGradTilingData* __restrict tiling_data = &tilingData;
  const TCubeTiling* __restrict matmulTilingWTiling = &(tiling_data->matmulTiling_w);
  const TCubeTiling* __restrict matmulTilingHTiling = &(tiling_data->matmulTiling_h);

  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  if (TILING_KEY_IS(1)) {
    UpSampleBicubic2dAAGradND<half> op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
    op.Init(input, output, userWS, &tilingData);
    op.Process();
  
  } else if (TILING_KEY_IS(2)) {
    UpSampleBicubic2dAAGradND<float> op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
    op.Init(input, output, userWS, &tilingData);
    op.Process();
  }else if (TILING_KEY_IS(3)) {
    UpSampleBicubic2dAAGradND<bfloat16_t> op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
    op.Init(input, output, userWS, &tilingData);
    op.Process();
  }
}