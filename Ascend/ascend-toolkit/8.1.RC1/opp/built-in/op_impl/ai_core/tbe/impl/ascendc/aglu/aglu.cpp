/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file aglu.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "aglu_weight_nz_l1_fullload.h"

using namespace AGLU;
using namespace AscendC;

extern "C" __global__ __aicore__ void aglu(GM_ADDR x, GM_ADDR weight1, GM_ADDR bias1,
                                           GM_ADDR weight2, GM_ADDR bias2, GM_ADDR y,
                                           GM_ADDR workspace, GM_ADDR tiling)
{
  if (workspace == nullptr) {
    return;
  }

  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);
#if __CCE_AICORE__ == 200
  if (TILING_KEY_IS(100)) {
    auto t = &tilingData;
    TPipe pipeOp;
    
    AGLUWeightNZL1Fullload op;
    op.Init(x, weight1, bias1, weight2, bias2, y, workspace, t, &pipeOp);
    op.Process();
  }
#endif

  return;
}