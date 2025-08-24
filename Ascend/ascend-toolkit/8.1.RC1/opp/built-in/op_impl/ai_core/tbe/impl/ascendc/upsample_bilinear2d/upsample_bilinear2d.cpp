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
 * \file upsample_bilinear2d.cpp
 * \brief
 */

#include "../upsample_linear1d/upsample_linear1d.h"

using namespace UpsampleLinear1d;

extern "C" __global__ __aicore__ void upsample_bilinear2d(GM_ADDR input, 
                                                        GM_ADDR size,
                                                        GM_ADDR output, 
                                                        GM_ADDR workspace, 
                                                        GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);

  const UpsampleLinear1dTilingData* __restrict tiling_data = &tilingData;
  const TCubeTiling* __restrict matmulTilingWTiling = &(tiling_data->matmulTiling_w);
  const TCubeTiling* __restrict matmulTilingHTiling = &(tiling_data->matmulTiling_h);

  // foreach(vector) not need workspace
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  if (TILING_KEY_IS(1)) {
    if(tiling_data->dataType == 1) {
      UpsampleLinear1dND<half> op;
      REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
      op.Init(input, output, userWS, &tilingData);
      op.Process();
    }
    if(tiling_data->dataType == 2) {
      UpsampleLinear1dND<float> op;
      REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
      op.Init(input, output, userWS, &tilingData);
      op.Process();
    }
    if(tiling_data->dataType == 3) {
      UpsampleLinear1dND<bfloat16_t> op;
      REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
      op.Init(input, output, userWS, &tilingData);
      op.Process();
    }
  }
}
