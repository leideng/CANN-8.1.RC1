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
 * \file upsample_bicubic2d_aa.cpp
 * \brief
 */

#include "upsample_bicubic2d_aa.h"

using namespace UpsampleBicubic2dAA;

extern "C" __global__ __aicore__ void upsample_bicubic2d_aa(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);

  const UpsampleBicubic2dAATilingData* __restrict tiling_data = &tilingData;
  const TCubeTiling* __restrict matmulTilingWTiling = &(tiling_data->matmulTilingW);
  const TCubeTiling* __restrict matmulTilingHTiling = &(tiling_data->matmulTilingH);

  // foreach(vector) not need workspace
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  if (TILING_KEY_IS(1)) {
    UpsampleBicubic2dAAND<half> op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
    op.Init(x, y, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(2)) {
    UpsampleBicubic2dAAND<float> op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
    op.Init(x, y, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(3)) {
    UpsampleBicubic2dAAND<bfloat16_t> op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
    op.Init(x, y, userWS, &tilingData);
    op.Process();
  }
}
