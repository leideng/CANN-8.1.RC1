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
 * \file group_norm_swish.cpp
 * \brief
 */

#include "group_norm_swish_hw1_b16.h"
#include "group_norm_swish_hw1_b32.h"
#include "group_norm_swish_small_b16.h"
#include "group_norm_swish_small_b32.h"
#include "group_norm_swish_norm_b16.h"
#include "group_norm_swish_norm_b32.h"
#include "group_norm_swish_large_b16.h"
#include "group_norm_swish_large_b32.h"

using namespace GroupNormSwish;

extern "C" __global__ __aicore__ void group_norm_swish(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                                      GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;

    if (TILING_KEY_IS(111)) {
        GroupNormSwish::GroupNormSwishHW1B16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(112)) {
        GroupNormSwish::GroupNormSwishSmallB16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(113)) {
        GroupNormSwish::GroupNormSwishNormB16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(114)) {
        GroupNormSwish::GroupNormSwishLargeB16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(121)) {
        GroupNormSwish::GroupNormSwishHW1B16<DTYPE_X, float> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(122)) {
        GroupNormSwish::GroupNormSwishSmallB16<DTYPE_X, float> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(123)) {
        GroupNormSwish::GroupNormSwishNormB16<DTYPE_X, float> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(124)) {
        GroupNormSwish::GroupNormSwishLargeB16<DTYPE_X, float> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(211)) {
        GroupNormSwish::GroupNormSwishHW1B32<float, float> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(212)) {
        GroupNormSwish::GroupNormSwishSmallB32<float, float> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(213)) {
        GroupNormSwish::GroupNormSwishNormB32<float, float> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(214)) {
        GroupNormSwish::GroupNormSwishLargeB32<float, float> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData, &pipe);
        op.Process();
    }
}
