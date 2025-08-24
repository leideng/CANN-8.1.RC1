/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
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
 * \file upsample_trilinear3d_backward.cpp
 * \brief
 */
#include "upsample_trilinear3d_backward.h"

using namespace UpsampleTrilinear3dBackward;

extern "C" __global__ __aicore__ void upsample_trilinear3d_backward(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    const UpsampleTrilinear3dBackwardTilingData *__restrict tilingData = &tiling_data;
    const TCubeTiling *__restrict matmulTilingW = &(tilingData->matmulTilingW);
    const TCubeTiling *__restrict matmulTilingH = &(tilingData->matmulTilingH);
    const TCubeTiling *__restrict matmulTilingD = &(tilingData->matmulTilingD);
    GM_ADDR userWs = GetUserWorkspace(workspace);
#define INIT_AND_PROCESS                          \
    REGIST_MATMUL_OBJ(&op.pipe,                   \
        GetSysWorkSpacePtr(),                     \
        op.matmulW,                               \
        matmulTilingW,                            \
        op.matmulH,                               \
        matmulTilingH,                            \
        op.matmulD,                               \
        matmulTilingD);                           \
    op.Init(input, output, userWs, &tiling_data); \
    op.Process()

    if (TILING_KEY_IS(1)) {
        if (tilingData->dataType == 1) {
            UpsampleTrilinear3dBackwardND<half> op;
            INIT_AND_PROCESS;
        } else if (tilingData->dataType == 2) {
            UpsampleTrilinear3dBackwardND<float> op;
            INIT_AND_PROCESS;
        } else if (tilingData->dataType == 3) {
            UpsampleTrilinear3dBackwardND<bfloat16_t> op;
            INIT_AND_PROCESS;
        }
    }
}
