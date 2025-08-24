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
 * \file foreach_copy.cpp
 * \brief
 */

#include "foreach_copy.h"

using namespace ForeachCopy;

extern "C" __global__ __aicore__ void foreach_copy(GM_ADDR x, GM_ADDR y, 
    GM_ADDR workspace, GM_ADDR tiling) {
    
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

#define INIT_AND_PROCESS                            \
    op.Init(x, y, userWS, &tilingData);       \
    op.Process()

    if (TILING_KEY_IS(1)) {
        ForeachCopyND<half> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2)) {
        ForeachCopyND<float> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3)) {
        ForeachCopyND<int> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(4)) {
        ForeachCopyND<bfloat16_t> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(5)) {
        ForeachCopyND<int16_t> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(6)) {
        ForeachCopyND<uint16_t> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(7)) {
        ForeachCopyND<int8_t> op;
        INIT_AND_PROCESS; 
    } else if (TILING_KEY_IS(8)) {
        ForeachCopyND<uint8_t> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(9)) {
        ForeachCopyND<uint32_t> op;
        INIT_AND_PROCESS; 
    } else if (TILING_KEY_IS(10)) {
        ForeachCopyND<uint64_t> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(11)) {
        ForeachCopyND<double> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(12)) {
        ForeachCopyND<bool> op;
        INIT_AND_PROCESS;
    }
}
