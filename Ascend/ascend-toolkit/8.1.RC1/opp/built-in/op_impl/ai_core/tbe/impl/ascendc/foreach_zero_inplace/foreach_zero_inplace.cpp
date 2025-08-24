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
 * \file foreach_zero_inplace.cpp
 * \brief
 */

#include "foreach_zero_inplace.h"

using namespace ForeachZeroInplace;

extern "C" __global__ __aicore__ void foreach_zero_inplace(GM_ADDR x,
    GM_ADDR workspace, GM_ADDR tiling) {
    
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachZeroInplaceND<half> op;
        op.Init(x, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachZeroInplaceND<float> op;
        op.Init(x, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        ForeachZeroInplaceND<int> op;
        op.Init(x, userWS, &tilingData);
        op.Process();
    } 
    #if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachZeroInplaceND<bfloat16_t> op;
        op.Init(x, userWS, &tilingData);
        op.Process();
    } 
    #endif
    else if (TILING_KEY_IS(5)) {
        ForeachZeroInplaceND<int16_t> op;
        op.Init(x, userWS, &tilingData);
        op.Process();
    }
}
