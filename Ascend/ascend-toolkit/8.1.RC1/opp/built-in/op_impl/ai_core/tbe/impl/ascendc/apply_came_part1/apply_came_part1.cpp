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
 * \file apply_came_part1.cpp
 * \brief
 */
 
#include "apply_came_part1_fp32.h"
#include "apply_came_part1_fp16.h"
#include "apply_came_part1_post.h"


using namespace ApplyCamePart1;

extern "C" __global__ __aicore__ void apply_came_part1(GM_ADDR grad, GM_ADDR eps, GM_ADDR sum_grad_r, GM_ADDR sum_grad_c, GM_ADDR sum_grad_rc, GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr)
    {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);


    if (TILING_KEY_IS(1001)) {
        ApplyCamePart1::ApplyCamePart1FP16<DTYPE_GRAD> op;
        op.Init(grad, eps, sum_grad_r, sum_grad_c, sum_grad_rc, userWS, &tilingData);
        op.Process();
        ApplyCamePart1::ApplyCamePart1Post<float> op_post;
        op_post.Init(grad, eps, sum_grad_r, sum_grad_c, sum_grad_rc, userWS, &tilingData);
        op_post.Process();
    } else if (TILING_KEY_IS(1002)) {
        ApplyCamePart1::ApplyCamePart1FP32<float> op;
        op.Init(grad, eps, sum_grad_r, sum_grad_c, sum_grad_rc, userWS, &tilingData);
        op.Process();
        ApplyCamePart1::ApplyCamePart1Post<float> op_post;
        op_post.Init(grad, eps, sum_grad_r, sum_grad_c, sum_grad_rc, userWS, &tilingData);
        op_post.Process();
    }
}
