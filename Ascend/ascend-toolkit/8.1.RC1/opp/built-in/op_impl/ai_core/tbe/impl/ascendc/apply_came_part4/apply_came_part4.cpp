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
 * \file apply_came_part4.cpp
 * \brief
 */
#include "apply_came_part4_float32.h"
#include "apply_came_part4_float16.h"
#include "apply_came_part4_pre.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void apply_came_part4(GM_ADDR paramIn, GM_ADDR m, GM_ADDR rIn, GM_ADDR cIn,
                                                       GM_ADDR weight_decay, GM_ADDR lr, GM_ADDR beta3, GM_ADDR sum_r,
                                                       GM_ADDR sum_u_r, GM_ADDR sum_u_c, GM_ADDR sum_u_rc,
                                                       GM_ADDR global_shape, GM_ADDR paramOut, GM_ADDR rOut,
                                                       GM_ADDR cOut, GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }

  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);

  if (TILING_KEY_IS(0)) {
    if (sum_r == nullptr) {
      ApplyCamePart4Pre<float> ApplyCamePart4Pre;
      ApplyCamePart4Pre.Init(rIn, userWS, &tilingData);
      ApplyCamePart4Pre.Process();
    }

    ApplyCamePart4Float32<float> ApplyCamePart4Float32;
    ApplyCamePart4Float32.Init(paramIn, m, rIn, cIn, weight_decay, lr, beta3, sum_r, sum_u_r, sum_u_c, sum_u_rc,
                               global_shape, paramOut, rOut, cOut, userWS, &tilingData);
    ApplyCamePart4Float32.Process();
  } else if (TILING_KEY_IS(1)) {
    if (sum_r == nullptr) {
      ApplyCamePart4Pre<DTYPE_M> ApplyCamePart4Pre;
      ApplyCamePart4Pre.Init(rIn, userWS, &tilingData);
      ApplyCamePart4Pre.Process();
    }

    ApplyCamePart4Float16<DTYPE_M> ApplyCamePart4Float16;
    ApplyCamePart4Float16.Init(paramIn, m, rIn, cIn, weight_decay, lr, beta3, sum_r, sum_u_r, sum_u_c, sum_u_rc,
                               global_shape, paramOut, rOut, cOut, userWS, &tilingData);
    ApplyCamePart4Float16.Process();
  }
}