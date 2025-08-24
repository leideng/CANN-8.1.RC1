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
 * \file flat_quant.cpp
 * \brief
 */

#include "flat_quant.h"

using namespace FlatQuantNS;

extern "C" __global__ __aicore__ void flat_quant(GM_ADDR x, GM_ADDR kronecker_p1, GM_ADDR kronecker_p2, GM_ADDR out, GM_ADDR quant_scale, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const FlatQuantTilingData *__restrict tiling_data = &tilingData;

    GM_ADDR userWS = GetUserWorkspace(workspace);

    if (TILING_KEY_IS(1)) {
        if ASCEND_IS_AIV{
            if (tiling_data->dataType == 1) {
                TestVec<half> vec;
                vec.Init(x, kronecker_p1, kronecker_p2, out, quant_scale, workspace, &tilingData);
                vec.Process();
            } else if (tiling_data->dataType == 2) {
                TestVec<bfloat16_t> vec;
                vec.Init(x, kronecker_p1, kronecker_p2, out, quant_scale, workspace, &tilingData);
                vec.Process();
            }
        }
        if ASCEND_IS_AIC{
            if (tiling_data->dataType == 1) {
                TestCube<half> cube;
                cube.Init(workspace, &tilingData);
                cube.Process();
            } else if (tiling_data->dataType == 2) {
                TestCube<bfloat16_t> cube;
                cube.Init(workspace, &tilingData);
                cube.Process();
            }
        }
    }
}
