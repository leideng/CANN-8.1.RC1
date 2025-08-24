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
 * \file foreach_round_off_number.cpp
 * \brief
 */
#include "foreach_round_off_number.h"

using namespace ForeachRoundOffNumber;

extern "C" __global__ __aicore__ void foreach_round_off_number(
    GM_ADDR x, GM_ADDR roundMode, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(2)) {
        ForeachRoundOffNumberND<float> op;
        op.Init(x, roundMode, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        ForeachRoundOffNumberND<half> op;
        op.Init(x, roundMode, y, userWS, &tilingData);
        op.Process();
    } 
    #if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachRoundOffNumberND<bfloat16_t> op;
        op.Init(x, roundMode, y, userWS, &tilingData);
        op.Process();
    }
    #endif
}
