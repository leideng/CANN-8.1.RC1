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
 * \file amp_update_scale.cpp
 * \brief
 */
#include "amp_update_scale.h"

extern "C" __global__ __aicore__ void amp_update_scale(GM_ADDR current_scale, GM_ADDR growth_tracker, GM_ADDR found_inf, GM_ADDR updated_scale, GM_ADDR updated_growth_tracker, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    AmpUpdateScale op;
    op.Init(current_scale, growth_tracker, found_inf, updated_scale, updated_growth_tracker, tiling_data.growthFactor, tiling_data.backoffFactor, tiling_data.growthInterval, &pipe);
    op.Process();
}