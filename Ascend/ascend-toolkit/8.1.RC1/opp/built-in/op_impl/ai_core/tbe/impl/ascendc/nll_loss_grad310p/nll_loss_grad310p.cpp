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
 * \file nll_loss_grad310p.cpp
 * \brief
 */

#include "nll_loss_grad310p.h"

using namespace KernelNLLLossGrad;

// kernel function
extern "C" __global__ __aicore__ void nll_loss_grad310p(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR target, GM_ADDR weight,
                              GM_ADDR totalweight,GM_ADDR out, 
                              GM_ADDR workspace, GM_ADDR tiling) {
  
  GET_TILING_DATA(tilingData, tiling);
  GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

  if (TILING_KEY_IS(2)) {
    KernelNLLLossGradND<half, int32_t> op;
    op.Init(gradOutput, self, target, weight, out, totalweight, &tilingData, usrWorkspace);
    op.Process();
  } 
  else if (TILING_KEY_IS(3)) {
    KernelNLLLossGradND<float, int32_t> op;
    op.Init(gradOutput, self, target, weight, out, totalweight, &tilingData, usrWorkspace);
    op.Process();
  }
}
