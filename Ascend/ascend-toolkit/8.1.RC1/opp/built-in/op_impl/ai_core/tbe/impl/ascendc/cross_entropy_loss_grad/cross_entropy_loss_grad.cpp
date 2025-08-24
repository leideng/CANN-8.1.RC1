/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file cross_entropy_loss_grad.cpp
 * \brief
 */

#include "cross_entropy_loss_grad_weight_not_none.h"
#include "cross_entropy_loss_grad_weight_none.h"

using namespace AscendC;
using namespace CrossEntropyLossGrad;
extern "C" __global__ __aicore__ void cross_entropy_loss_grad(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target,
                                                              GM_ADDR weight, GM_ADDR grad_zloss, GM_ADDR lse_for_zloss,
                                                              GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
  // 10: bf16, 不存在weight
  // 11: bf16, 存在weight
  // 20: fp16, 不存在weight
  // 21: fp16, 存在weight
  // 30: fp32, 不存在weight
  // 31: fp32, 存在weight
  if (TILING_KEY_IS(10)) {
    CrossEntropyLossGradWeightNone<bfloat16_t> CrossEntropyLossGradWeightNoneOp;
    CrossEntropyLossGradWeightNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNoneOp.Process();
  }
  else if (TILING_KEY_IS(11)) {
    CrossEntropyLossGradWeightNotNone<bfloat16_t> CrossEntropyLossGradWeightNotNoneOp;
    CrossEntropyLossGradWeightNotNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNotNoneOp.Process();
  }
  else if (TILING_KEY_IS(20)) {
    CrossEntropyLossGradWeightNone<half> CrossEntropyLossGradWeightNoneOp;
    CrossEntropyLossGradWeightNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNoneOp.Process();
  }
  else if (TILING_KEY_IS(21)) {
    CrossEntropyLossGradWeightNotNone<half> CrossEntropyLossGradWeightNotNoneOp;
    CrossEntropyLossGradWeightNotNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNotNoneOp.Process();
  }
  else if (TILING_KEY_IS(30)) {
    CrossEntropyLossGradWeightNone<float> CrossEntropyLossGradWeightNoneOp;
    CrossEntropyLossGradWeightNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNoneOp.Process();
  }
  else if (TILING_KEY_IS(31)) {
    CrossEntropyLossGradWeightNotNone<float> CrossEntropyLossGradWeightNotNoneOp;
    CrossEntropyLossGradWeightNotNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNotNoneOp.Process();
  }
}