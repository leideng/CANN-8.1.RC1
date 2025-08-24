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
 * \file group_norm_grad.cpp
 * \brief
 */
#include "group_norm_grad.h"

extern "C" __global__ __aicore__ void group_norm_grad(
    GM_ADDR dy, 
    GM_ADDR mean, 
    GM_ADDR rstd, 
    GM_ADDR x, 
    GM_ADDR gamma, 
    GM_ADDR dx, 
    GM_ADDR dgamma, 
    GM_ADDR dbeta, 
    GM_ADDR workspace, 
    GM_ADDR tilingdata) {
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tilingdata);
    if (TILING_KEY_IS(0)) {
        GroupNormGrad<float, false> opFP32(
            dy, mean, rstd, x, gamma,
            dx, dgamma, dbeta, usrWorkspace,
            &tiling_data);
        opFP32.Process();
    } else if (TILING_KEY_IS(1)) {
        GroupNormGrad<half, false> opFP16(
            dy, mean, rstd, x, gamma,
            dx, dgamma, dbeta, usrWorkspace,
            &tiling_data);
        opFP16.Process();
    } else if (TILING_KEY_IS(2)) {
        GroupNormGrad<bfloat16_t, false> opBF16(
            dy, mean, rstd, x, gamma,
            dx, dgamma, dbeta, usrWorkspace,
            &tiling_data);
        opBF16.Process();
    } else if (TILING_KEY_IS(10)) {
        GroupNormGrad<float, true> opFP32Det(
            dy, mean, rstd, x, gamma,
            dx, dgamma, dbeta, usrWorkspace,
            &tiling_data);
        opFP32Det.Process();
    } else if (TILING_KEY_IS(11)) {
        GroupNormGrad<half, true> opFP16Det(
            dy, mean, rstd, x, gamma,
            dx, dgamma, dbeta, usrWorkspace,
            &tiling_data);
        opFP16Det.Process();
    } else if (TILING_KEY_IS(12)) {
        GroupNormGrad<bfloat16_t, true> opBF16Det(
            dy, mean, rstd, x, gamma,
            dx, dgamma, dbeta, usrWorkspace,
            &tiling_data);
        opBF16Det.Process();
    }
}