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
 * \file foreach_minimum_list.cpp
 * \brief
 */
#include "kernel_operator.h"

// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "../foreach_utils/foreach_no_scalar_binary.h"

using namespace AscendC;
using namespace Common::OpKernel;

extern "C" __global__ __aicore__ void foreach_minimum_list(GM_ADDR inputs_1, GM_ADDR inputs_2,
    GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachNoScalarBinary<half, half, Min> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachNoScalarBinary<float, float, Min> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        ForeachNoScalarBinary<int, int, Min> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    }  
    #if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachNoScalarBinary<bfloat16_t, float, Min> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    }
    #endif
}
