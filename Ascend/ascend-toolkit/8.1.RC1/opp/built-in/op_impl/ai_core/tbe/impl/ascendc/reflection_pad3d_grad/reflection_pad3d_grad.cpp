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
 * \file reflection_pad3d_grad.cpp
 * \brief
 */
#include "reflection_pad3d_grad_mid.h"
#include "reflection_pad3d_grad_small.h"

extern "C" __global__ __aicore__ void reflection_pad3d_grad(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    if (TILING_KEY_IS(0)) {
        ReflectionPad3dGrad<float> op;
        op.Init(tiling_data, x, paddings, y, userWS);
        op.SmallProcess();
    } else if(TILING_KEY_IS(1)) {
        ReflectionPad3dGrad<float> op;
        op.Init(tiling_data, x, paddings, y, userWS);
        op.MidProcess();
    } else if(TILING_KEY_IS(2)) {
        ReflectionPad3dGrad<half> op;
        op.Init(tiling_data, x, paddings, y, userWS);
        op.SmallProcess();
    } else if(TILING_KEY_IS(3)) {
        ReflectionPad3dGrad<half> op;
        op.Init(tiling_data, x, paddings, y, userWS);
        op.MidProcess();
    } else if(TILING_KEY_IS(4)) {
        ReflectionPad3dGrad<bfloat16_t> op;
        op.Init(tiling_data, x, paddings, y, userWS);
        op.SmallProcess();
    } else if(TILING_KEY_IS(5)) {
        ReflectionPad3dGrad<bfloat16_t> op;
        op.Init(tiling_data, x, paddings, y, userWS);
        op.MidProcess();
    }
}