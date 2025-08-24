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
 * \file moe_init_routing_v2_grad.cpp
 * \brief
 */
#include "moe_init_routing_v2_grad_with_dropless.h"
#include "moe_init_routing_v2_grad_with_activate.h"
#include "moe_init_routing_v2_grad_with_pos_drop_and_pad_zero.h"

using namespace AscendC;
using namespace MoeInitRoutingV2Grad;

#define MOE_INIT_ROUTING_V2_GRAD_DROPLESS_FLOAT        1000UL
#define MOE_INIT_ROUTING_V2_GRAD_DROPLESS_FLOAT16      1001UL
#define MOE_INIT_ROUTING_V2_GRAD_DROPLESS_BF16         1002UL
#define MOE_INIT_ROUTING_V2_GRAD_ACTIVATE_FLOAT        1010UL
#define MOE_INIT_ROUTING_V2_GRAD_ACTIVATE_FLOAT16      1011UL
#define MOE_INIT_ROUTING_V2_GRAD_ACTIVATE_BF16         1012UL
#define MOE_INIT_ROUTING_V2_GRAD_POSITION_PAD0_FLOAT   1100UL
#define MOE_INIT_ROUTING_V2_GRAD_POSITION_PAD0_FLOAT16 1101UL
#define MOE_INIT_ROUTING_V2_GRAD_POSITION_PAD0_BF16    1102UL

extern "C" __global__ __aicore__ void moe_init_routing_v2_grad(GM_ADDR gradExpandedX, GM_ADDR expandedRowIdx,
                                                               GM_ADDR gradX, GM_ADDR workspace, GM_ADDR tiling) {
  if (g_coreType == AIC) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);

  TPipe pipeOp;
  if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_DROPLESS_FLOAT)) {
    MoeInitRoutingV2GradDroplessCompute<DTYPE_GRAD_EXPANDED_X> op;
    op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_DROPLESS_FLOAT16)) {
    MoeInitRoutingV2GradDroplessCompute<DTYPE_GRAD_EXPANDED_X> op;
    op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_DROPLESS_BF16)) {
    MoeInitRoutingV2GradDroplessCompute<DTYPE_GRAD_EXPANDED_X> op;
    op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_ACTIVATE_FLOAT)) {
    MoeInitRoutingV2GradActivateCompute<DTYPE_GRAD_EXPANDED_X> op;
    op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_ACTIVATE_FLOAT16)) {
    MoeInitRoutingV2GradActivateCompute<DTYPE_GRAD_EXPANDED_X> op;
    op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_ACTIVATE_BF16)) {
    MoeInitRoutingV2GradActivateCompute<DTYPE_GRAD_EXPANDED_X> op;
    op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_POSITION_PAD0_FLOAT)) {
    MoeInitRoutingV2GradPositionPad0Compute<DTYPE_GRAD_EXPANDED_X> op;
    op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_POSITION_PAD0_FLOAT16)) {
    MoeInitRoutingV2GradPositionPad0Compute<DTYPE_GRAD_EXPANDED_X> op;
    op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_POSITION_PAD0_BF16)) {
    MoeInitRoutingV2GradPositionPad0Compute<DTYPE_GRAD_EXPANDED_X> op;
    op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
    op.Process();
  }
  pipeOp.Destroy();
}
