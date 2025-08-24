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
 * \file moe_tutel_dispatch.cpp
 * \brief
 */
#include "moe_tutel_dispatch.h"

extern "C" __global__ __aicore__ void moe_tutel_dispatch(GM_ADDR x, GM_ADDR gates, GM_ADDR indices,
                                                            GM_ADDR locations, GM_ADDR y,
                                                            GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        MoeTutelDispatch<float> op;
        op.Init(x, gates, indices, locations, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        MoeTutelDispatch<half> op;
        op.Init(x, gates, indices, locations, y, &tilingData);
        op.Process();
    }
}
