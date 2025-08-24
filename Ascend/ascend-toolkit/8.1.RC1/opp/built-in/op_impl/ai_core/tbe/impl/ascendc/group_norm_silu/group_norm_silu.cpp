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
 * \file group_norm_silu.cpp
 * \brief
 */
#include "group_norm_silu_b16.h"
#include "group_norm_silu_b32.h"
#include "group_norm_silu_g16.h"
#include "group_norm_silu_g32.h"
#include "group_norm_silu_small_b16.h"
#include "group_norm_silu_small_b32.h"
#include "group_norm_silu_hw1_b16.h"
#include "group_norm_silu_hw1_b32.h"
#include "group_norm_silu_sd.h"

using namespace GroupNormSilu;

extern "C" __global__ __aicore__ void group_norm_silu(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu,
                                                      GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1011)) {
        GroupNormSilu::GroupNormSiluSmallB16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1012)) {
        GroupNormSilu::GroupNormSiluSmallB16<DTYPE_X, float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(102)) {
        GroupNormSilu::GroupNormSiluSmallB32<float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1031)) {
        GroupNormSilu::GroupNormSiluB16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1032)) {
        GroupNormSilu::GroupNormSiluB16<DTYPE_X, float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(104)) {
        GroupNormSilu::GroupNormSiluB32<float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1051)) {
        GroupNormSilu::GroupNormSiluG16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1052)) {
        GroupNormSilu::GroupNormSiluG16<DTYPE_X, float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(106)) {
        GroupNormSilu::GroupNormSiluG32<float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1071)) {
        GroupNormSilu::GroupNormSiluHW1B16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1072)) {
        GroupNormSilu::GroupNormSiluHW1B16<DTYPE_X, float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(108)) {
        GroupNormSilu::GroupNormSiluHW1B32<float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(109)) {
        GroupNormSilu::GroupNormSiluSD<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } 
}
