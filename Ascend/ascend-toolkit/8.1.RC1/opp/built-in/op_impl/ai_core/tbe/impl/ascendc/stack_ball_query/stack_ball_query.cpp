/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file stack_ball_query.cpp
 * \brief
 */
#include "stack_ball_query.h"

extern "C" __global__ __aicore__ void stack_ball_query(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR xyz_batch_cnt,
                      GM_ADDR center_xyz_batch_cnt, GM_ADDR idx, GM_ADDR workspace,
                      GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KernelStackBallQuery<float> op;
        op.Init(xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, idx, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelStackBallQuery<half> op;
        op.Init(xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, idx, tilingData);
        op.Process();
    }
}