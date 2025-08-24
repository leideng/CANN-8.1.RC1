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
 * \file moe_gating_top_k.cpp
 * \brief
 */

#include "moe_gating_top_k_e_k_fullload.h"
using namespace AscendC;
using namespace MoeGatingTopK;
extern "C" __global__ __aicore__ void moe_gating_top_k(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx,
                                                       GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
  if (g_coreType == AIC) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);
  if (workspace == nullptr) {
    return;
  }

  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  auto t = &tilingData;
  TPipe tPipe;
  if (TILING_KEY_IS(1)) {
    MoeGatingTopKEKFullload<DTYPE_X> op;
    op.Init(x, bias, y, expertIdx, out, userWS, t, &tPipe);
    op.Process();
  }
}
