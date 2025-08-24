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
 * \file grouped_matmul_finalize_routing.cpp
 * \brief
 */
#include "grouped_matmul_finalize_routing.h"

using namespace AscendC;
using namespace matmul;
using namespace GroupedMatmulFinalizeRouting;

extern "C" __global__ __aicore__ void grouped_matmul_finalize_routing(GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias,
                                                                      GM_ADDR pertoken_scale, GM_ADDR group_list,
                                                                      GM_ADDR share_input, GM_ADDR logit,
                                                                      GM_ADDR row_index, GM_ADDR y, GM_ADDR workspaceGM,
                                                                      GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(10000000000000000001UL)) {
        TPipe pipe;
        MT mm;
        if ASCEND_IS_AIC {
            mm.SetSubBlockIdx(0);
            mm.Init(&tilingData.matmulTiling, &pipe);
        }
        using param = Param<true, GroupMatmulFRTilingData>;
        QuantGroupMatmul<param> op(mm);
        op.Init(x, w, bias, group_list, scale, pertoken_scale, logit, row_index, share_input, y, workspaceGM,
                &tilingData, &pipe);
        op.Process();
    }
}
