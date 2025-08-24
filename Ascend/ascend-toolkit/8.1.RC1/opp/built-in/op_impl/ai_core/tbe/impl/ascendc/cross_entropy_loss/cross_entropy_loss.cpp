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
 * \file cross_entropy_loss.cpp
 * \brief
 */

#include "cross_entropy_loss.h"
#include "cross_entropy_loss_fp32.h"

using namespace CrossEntropyLossCustom;

extern "C" __global__ __aicore__ void cross_entropy_loss(GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob, 
                                                         GM_ADDR zloss, GM_ADDR lse_for_zloss, GM_ADDR workspace, GM_ADDR tiling) 
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    if (TILING_KEY_IS(1)) {
        CrossEntropyLoss<bfloat16_t> crossEntropyLossOp;
        crossEntropyLossOp.Init(input, target, weight, loss, log_prob, zloss, lse_for_zloss, usrWorkspace, tilingData);
        crossEntropyLossOp.Process();
    }

    if (TILING_KEY_IS(2)) {
        CrossEntropyLoss<half> crossEntropyLossOp;
        crossEntropyLossOp.Init(input, target, weight, loss, log_prob, zloss, lse_for_zloss, usrWorkspace, tilingData);
        crossEntropyLossOp.Process();
    }

    if (TILING_KEY_IS(3)) {
        CrossEntropyLoss<float> crossEntropyLossOp;
        crossEntropyLossOp.Init(input, target, weight, loss, log_prob, zloss, lse_for_zloss, usrWorkspace, tilingData);
        crossEntropyLossOp.ProcessFp32();
    }
}
    
