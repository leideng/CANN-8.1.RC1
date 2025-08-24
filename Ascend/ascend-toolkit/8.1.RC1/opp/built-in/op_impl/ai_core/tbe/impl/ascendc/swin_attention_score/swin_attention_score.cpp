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
 * \file swin_attention_score.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "swin_attention_score_split_b_n.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void swin_attention_score(GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR input_mask1, GM_ADDR input_mask2, GM_ADDR sm, GM_ADDR dm, GM_ADDR attention_score, GM_ADDR softmax_out, GM_ADDR workspace, GM_ADDR tiling2) {
    TPipe tPipe;
    set_mask_norm();

    // get tiling data
    GET_TILING_DATA(tiling_data, tiling2);

    // get workspace
    __gm__ uint8_t* user = GetUserWorkspace(workspace);

    swin::SwinAttentionScore<half, half> op;
    op.Init(query, key, value, input_mask1, sm, attention_score, user, &tiling_data, &tPipe);
    op.Process();
}