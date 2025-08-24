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
 * \file avg_pool3_d_grad.cpp
 * \brief
 */
#include "avg_pool3_d_grad_no_cast.h"
#include "avg_pool3_d_grad_cast.h"
#include "avg_pool3_d_grad_t.h"
#include "avg_pool3_d_grad_t_cast.h"

using namespace AvgPool3DGrad;

extern "C" __global__ __aicore__ void avg_pool3_d_grad(GM_ADDR orig_input_shape, GM_ADDR grads, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tiling);
    #define INIT_AND_PROCESS                \
        op.Init(grads, output, tiling_data, userWorkspace);\
        op.Process()

    if (TILING_KEY_IS(1000)) {
        AvgPool3DGrad::KernelAvgPool3DGradCast<DTYPE_GRADS> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2000)) {
        AvgPool3DGrad::KernelAvgPool3DGradNoCast<float> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3000)) {
        AvgPool3DGrad::KernelAvgPool3DGradT<float> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(4000)) {
        AvgPool3DGrad::KernelAvgPool3DGradTCast<DTYPE_GRADS> op;
        INIT_AND_PROCESS;
    }
}