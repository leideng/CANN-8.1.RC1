/* *
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

/* !
 * \file moe_finalize_routing_v2_grad.cpp
 * \brief
 */
#include "moe_finalize_routing_v2_grad_with_scale_cut_h.h"
#include "moe_finalize_routing_v2_grad_with_scale_not_cut_h.h"
#include "moe_finalize_routing_v2_grad_without_scale_cut_h.h"
#include "moe_finalize_routing_v2_grad_without_scale_not_cut_h.h"

#define TILING_KEY_WITHOUT_SCALE_NOT_CUT_H 10001
#define TILING_KEY_WITHOUT_SCALE_CUT_H 10002
#define TILING_KEY_WITH_SCALE_NOT_CUT_H 20001
#define TILING_KEY_WITH_SCALE_CUT_H 20002

using namespace MoeFinalizeRoutingV2Grad;

extern "C" __global__ __aicore__ void moe_finalize_routing_v2_grad(GM_ADDR gradY, GM_ADDR expandedRowIdx,
    GM_ADDR expandedX, GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR bias, GM_ADDR gradExpandedX, GM_ADDR gradScales,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    TPipe pipe;

    if (TILING_KEY_IS(TILING_KEY_WITHOUT_SCALE_NOT_CUT_H)) {
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradWithoutScaleNotCutH<DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX> op;
        op.Init(gradY, expandedRowIdx, gradExpandedX, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITHOUT_SCALE_CUT_H)) {
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradWithoutScaleCutH<DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX> op;
        op.Init(gradY, expandedRowIdx, gradExpandedX, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITH_SCALE_NOT_CUT_H)) {
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradWithScaleNotCutH<DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX> op;
        op.Init(gradY, expandedRowIdx, expandedX, scales, expertIdx, bias, gradExpandedX, gradScales, workspace,
            &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITH_SCALE_CUT_H)) {
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradWithScaleCutH<DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX> op;
        op.Init(gradY, expandedRowIdx, expandedX, scales, expertIdx, bias, gradExpandedX, gradScales, workspace,
            &tilingData, &pipe);
        op.Process();
    } else {
        return;
    }
}