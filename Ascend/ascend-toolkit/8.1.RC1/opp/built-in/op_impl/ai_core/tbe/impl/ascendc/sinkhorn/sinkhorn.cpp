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
 * \file sinkhorn.cpp
 * \brief
 */
#include "sinkhorn.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void sinkhorn(GM_ADDR cost, GM_ADDR p, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    // 这两句都不要删除，有些调用框架还不能适配，需要保留
    SetSysWorkspace(workspace); 
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace); // 获取用户workspace指针。

    if (TILING_KEY_IS(0)) {
        // ge::DT_FLOAT
        AscendC::KernelSinkhorn<float, float> op;
        op.Init(cost, p, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        // ge::DT_FLOAT16
        AscendC::KernelSinkhorn<half, half> op;
        op.Init(cost, p, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(27)) {
        // ge::DT_BFLOAT16
        AscendC::KernelSinkhorn<float, bfloat16_t> op;
        op.Init(cost, p, usrWorkspace, &tiling_data);
        op.Process();
    }
}
