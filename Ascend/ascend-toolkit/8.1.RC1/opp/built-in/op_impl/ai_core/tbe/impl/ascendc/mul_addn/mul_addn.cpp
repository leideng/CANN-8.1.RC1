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
 * \file mul_addn.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "mul_addn_align.h"
#include "mul_addn_align_bf16.h"

extern "C" __global__ __aicore__ void mul_addn(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        TPipe pipe;
        KernelMulAddnAlign<float> op;
        op.Init(x1, x2, y, workspace, &tilingData, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.Process();
        op.ReleaseEventID();
    }else if (TILING_KEY_IS(1)) {
        TPipe pipe;
        KernelMulAddnAlignF16<half, half> op;
        op.Init(x1, x2, y, workspace, &tilingData, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.Process();
        op.ReleaseEventID();
    }else if (TILING_KEY_IS(2)) {
        TPipe pipe;
        KernelMulAddnAlignF16<bfloat16_t, float> op;
        op.Init(x1, x2, y, workspace, &tilingData, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.Process();
        op.ReleaseEventID();
    } 

}