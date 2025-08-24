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
 * \file pad_v3_grad_replication.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "pad_v3_grad_replication.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void pad_v3_grad_replication(GM_ADDR x, GM_ADDR paddings, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        PadV3GradReplication<float, false> op(tilingData);
        op.Init(x, z, workspace);
        op.Process();
        return;
    } else if (TILING_KEY_IS(2)) {
        PadV3GradReplication<half, true> op(tilingData);
        op.Init(x, z, workspace);
        op.Process();
        return;
    } else if (TILING_KEY_IS(3)) {
        PadV3GradReplication<bfloat16_t, true> op(tilingData);
        op.Init(x, z, workspace);
        op.Process();
        return;
    }
}