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
 * \file fatrelu_mul.cpp
 * \brief fatrelu_mul kernel
 */

#include "fatrelu_mul.h"

using namespace AscendC;

using namespace FatreluMul;

extern "C" __global__ __aicore__ void fatrelu_mul(GM_ADDR input,
                                                  GM_ADDR scalar,
                                                  GM_ADDR output,
                                                  GM_ADDR workspace,
                                                  GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWs = nullptr;

#if __CCE_AICORE__ == 220
    if (TILING_KEY_IS(1)) {
        FatreluMulND<half> op;
        op.Init(input, scalar, output, userWs, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        FatreluMulND<float> op;
        op.Init(input, scalar, output, userWs, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        FatreluMulND<bfloat16_t> op;
        op.Init(input, scalar, output, userWs, &tilingData);
        op.Process();
    }
#else
#endif
}