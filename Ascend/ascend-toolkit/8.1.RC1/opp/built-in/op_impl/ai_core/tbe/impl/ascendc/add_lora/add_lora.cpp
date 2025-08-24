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
 * \file add_lora.cpp
 * \brief
 */


#if __CCE_AICORE__ == 200
#include "add_lora_310.h"
#include "add_lora_sparse_310.h"
#include "bgmv_310.h"
#else
#include "add_lora_single_core.h"
#include "add_lora_normal_core.h"
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void add_lora(
    GM_ADDR y,
    GM_ADDR x,
    GM_ADDR weightB,
    GM_ADDR indices,
    GM_ADDR weightA,
    GM_ADDR y_out,
    GM_ADDR workspace,
    GM_ADDR tiling
){
    if (workspace == nullptr) {
        return;
    }
    TPipe tPipe;
    GM_ADDR user1 = GetUserWorkspace(workspace);
    if (user1 == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    #if __CCE_AICORE__ == 200
        if (TILING_KEY_IS(200000)) {
            AddLoraKernel310 op;
            op.Init(y, x, weightA, weightB, indices, y_out, user1, tilingData, &tPipe);
            op.Process();
            tPipe.Destroy();
        } else if (TILING_KEY_IS(200001)) {
            AddLoraSparse310 op;
            op.Init(y, x, weightA, weightB, indices, y_out, user1, tilingData, &tPipe);
            op.Process();
            tPipe.Destroy();
        } else if (TILING_KEY_IS(200010)) {
            BgmvKernel310 op;
            op.Init(y, x, weightA, weightB, indices, y_out, user1, tilingData, &tPipe);
            op.Process();
            tPipe.Destroy();
        }
    #else
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        if (TILING_KEY_IS(100000)) {
            AddLoraNormalCoreBatchKernel op;
            op.Init(y, x, weightB, indices, weightA, y_out, user1, tilingData, &tPipe);
            op.Process();
            tPipe.Destroy();
        } else if (TILING_KEY_IS(100001)) {
            AddLoraSingleCoreBatchKernel op;
            op.Init(y, x, weightB, indices, weightA, y_out, user1, tilingData, &tPipe);
            op.Process();
            tPipe.Destroy();
        }
    #endif
}