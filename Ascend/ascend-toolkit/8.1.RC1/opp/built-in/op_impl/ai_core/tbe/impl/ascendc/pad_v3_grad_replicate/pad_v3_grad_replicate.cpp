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
 * \file pad_v3_grad_replicate.cpp
 * \brief
 */

#include "pad_v3_grad_replicate_h.h"
#include "pad_v3_grad_replicate_w.h"
#include "pad_v3_grad_replicate_h_w.h"
#include "pad_v3_grad_replicate_h_w_mini.h"
#include "pad_v3_grad_replicate_small_h_large_w.h"
#include "pad_v3_grad_replicate_small_h_large_w_bf16.h"
#include "pad_v3_grad_replicate_large_h_small_w.h"
#include "pad_v3_grad_replicate_large_h_small_w_bf16.h"
#include "pad_v3_grad_replicate_h_w_large.h"

extern "C" __global__ __aicore__ void pad_v3_grad_replicate(GM_ADDR x, GM_ADDR paddings, GM_ADDR y,
                                                            GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1110)) {
        PadV3GradReplicateH<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2110)) {
        PadV3GradReplicateH<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3110)) {
        PadV3GradReplicateH<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1101)) {
        PadV3GradReplicateW<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2101)) {
        PadV3GradReplicateW<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3101)) {
        PadV3GradReplicateW<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1000)) {
        PadV3GradReplicateHWMini<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2000)) {
        PadV3GradReplicateHWMini<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3000)) {
        PadV3GradReplicateHWMini<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1100)) {
        PadV3GradReplicateSmallHLargeW<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2100)) {
        PadV3GradReplicateSmallHLargeW<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3100)) {
        PadV3GradReplicateSmallHLargeWBf16<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1010)) {
        PadV3GradReplicateLargeHSmallW<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2010)) {
        PadV3GradReplicateLargeHSmallW<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3010)) {
        PadV3GradReplicateLargeHSmallWBf16<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1111)) {
        PadV3GradReplicateHWLarge<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2111)) {
        PadV3GradReplicateHWLarge<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3111)) {
        PadV3GradReplicateHWLarge<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(11111)) {
        PadV3GradReplicateHW<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(22222)) {
        PadV3GradReplicateHW<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(33333)) {
        PadV3GradReplicateHW<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
}