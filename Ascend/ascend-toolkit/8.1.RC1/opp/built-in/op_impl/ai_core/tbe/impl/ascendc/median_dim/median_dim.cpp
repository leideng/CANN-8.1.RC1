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
 * \file median_dim.cpp
 * \brief
 */
#include "median_dim.h"

using namespace AscendC;
    // kernel function
extern "C" __global__ __aicore__ void median_dim(GM_ADDR self, GM_ADDR values, GM_ADDR indices, 
                                                 GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR gmTensor[3] = {self, values, indices};
    if (TILING_KEY_IS(1)) {
        MedianDimKernel<float, int> op(gmTensor, tilingData, &pipe);
        op.Process();
    } 
}