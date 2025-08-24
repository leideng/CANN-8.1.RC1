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
 * \file conv2d_backprop_filter_v2.cpp
 * \brief
 */
#include "conv2d_backprop_filter_v2.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void conv2d_backprop_filter_v2(GM_ADDR x, GM_ADDR filter_size, GM_ADDR out_backprop,
                                                                GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
    if (workSpace == nullptr) {
        return;
    }

    GM_ADDR user1 = GetUserWorkspace(workSpace);
    if (user1 == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        Conv2dDw<DTYPE_X, FORMAT_X, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, FORMAT_Y> op;
        op.Init(x, out_backprop, y, user1, &tilingData);
        op.Process();
    }
}