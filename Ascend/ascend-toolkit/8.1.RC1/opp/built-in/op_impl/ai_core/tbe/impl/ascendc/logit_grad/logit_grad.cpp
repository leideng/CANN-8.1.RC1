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
 * \file logit_grad.cpp
 * \brief logit_grad kernel
 */

#include "logit_grad.h"

using namespace AscendC;

using namespace LogitGrad;

extern "C" __global__ __aicore__ void logit_grad(GM_ADDR x,
                                                 GM_ADDR dy,
                                                 GM_ADDR dx,
                                                 GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWs = nullptr;

#if __CCE_AICORE__ == 220
    if (TILING_KEY_IS(1)) {
        LogitGradND<half> op;
        op.Init(x, dy, dx, userWs, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        LogitGradND<float> op;
        op.Init(x, dy, dx, userWs, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        LogitGradND<bfloat16_t> op;
        op.Init(x, dy, dx, userWs, &tilingData);
        op.Process();
    }
#else
#endif
}