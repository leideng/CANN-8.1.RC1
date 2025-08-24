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
 * \file upsample_nearest3d.cpp
 * \brief
 */

#if __CCE_AICORE__ == 200
#include "upsample_nearest3d_310p.h"
#else
#include "upsample_nearest3d.h"
#endif

using namespace UpsampleNearest3d;

extern "C" __global__ __aicore__ void upsample_nearest3d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  const UpsampleNearest3dTilingData* __restrict tiling_data = &tilingData;

  GM_ADDR userWS = GetUserWorkspace(workspace);

#define INIT_AND_PROCESS                     \
  op.Init(x, y, false, userWS, &tilingData); \
  op.Process()

#if __CCE_AICORE__ == 200
  if (TILING_KEY_IS(1)) {
    if (tiling_data->dataType == 1) {
      UpsampleNearest3dND310p<half> op;
      INIT_AND_PROCESS;
    } else if (tiling_data->dataType == 2) {
      UpsampleNearest3dND310p<float> op;
      INIT_AND_PROCESS;
    }
  }
#else
  if (TILING_KEY_IS(1)) {
    if (tiling_data->dataType == 1) {
      UpsampleNearest3dND<half> op;
      INIT_AND_PROCESS;
    } else if (tiling_data->dataType == 2) {
      UpsampleNearest3dND<float> op;
      INIT_AND_PROCESS;
    } else if (tiling_data->dataType == 3) {
      UpsampleNearest3dND<bfloat16_t> op;
      INIT_AND_PROCESS;
    }
  }
#endif
}
