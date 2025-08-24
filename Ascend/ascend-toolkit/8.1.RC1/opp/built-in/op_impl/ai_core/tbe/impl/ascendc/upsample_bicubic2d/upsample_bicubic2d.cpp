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
 * \file upsample_bicubic2d.cpp
 * \brief
 */

#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 300
#include "upsample_bicubic2d_310p.h"
#else
#include "upsample_bicubic2d.h"
#endif

using namespace UpsampleBicubic2d;

extern "C" __global__ __aicore__ void upsample_bicubic2d(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                         GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);

  const UpsampleBicubic2dTilingData* __restrict tiling_data = &tilingData;
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 300
  if (TILING_KEY_IS(1)) {
    if (tiling_data->dataType == 1) {
      UpsampleBicubic2dND310p<half> op;
      op.Init(input, output, userWS, &tilingData);
      op.Process();
    }
    if (tiling_data->dataType == 2) {
      UpsampleBicubic2dND310p<float> op;
      op.Init(input, output, userWS, &tilingData);
      op.Process();
    }
  }
#else
  const TCubeTiling* __restrict matmulTilingWTiling = &(tiling_data->matmulTiling_w);
  const TCubeTiling* __restrict matmulTilingHTiling = &(tiling_data->matmulTiling_h);

  if (TILING_KEY_IS(1)) {
    if (tiling_data->dataType == 1) {
      UpsampleBicubic2dND<half> op;
      REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH,
                        matmulTilingHTiling);
      op.Init(input, output, userWS, &tilingData);
      op.Process();
    }
    if (tiling_data->dataType == 2) {
      UpsampleBicubic2dND<float> op;
      REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH,
                        matmulTilingHTiling);
      op.Init(input, output, userWS, &tilingData);
      op.Process();
    }
    if (tiling_data->dataType == 3) {
      UpsampleBicubic2dND<bfloat16_t> op;
      REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH,
                        matmulTilingHTiling);
      op.Init(input, output, userWS, &tilingData);
      op.Process();
    }
  }
#endif
}
