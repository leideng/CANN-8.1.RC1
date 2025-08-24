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
 * \file foreach_norm.cpp
 * \brief
 */

#include "foreach_norm.h"

using namespace ForeachNorm;

/**
 * modelCode:
 * 0 p=else... the default operator(not used now)
 * 1 p=0 Calculate the count of not-zero element in each tensor. (not used now)
 * 2 p=1 AbsAndNotNeedPower NotNeedSqrt(now is default as we now only consider p=1 || p=2)
 * 3 p=2 MulSelf(we don't need abs this time) Sqrt(not Power(self,1/p))
 * 4 p=+inf Calculate the max Abs(element) in each tensor. (not used now)
 * 5 p=-inf Calculate the min Abs(element) in each tensor. (not used now)
 */

constexpr uint8_t ZERO_SCALAR_NORM_MODEL_CODE = 0;
constexpr uint8_t ONE_SCALAR_NORM_MODEL_CODE = 1;
constexpr uint8_t TWO_SCALAR_NORM_MODEL_CODE = 2;
constexpr uint8_t POSITIVE_INF_SCALAR_NORM_MODEL_CODE = 3;
constexpr uint8_t NEGATIVE_INF_SCALAR_NORM_MODEL_CODE = 4;
constexpr uint8_t DEFAULT_SCALAR_NORM_MODEL_CODE = 5;

extern "C" __global__ __aicore__ void foreach_norm(GM_ADDR inputs, GM_ADDR scalar, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);

    uint8_t modelCode = TWO_SCALAR_NORM_MODEL_CODE;
    GlobalTensor<DTYPE_SCALAR> inScalarGM;
    inScalarGM.SetGlobalBuffer((__gm__ DTYPE_SCALAR*)scalar, 1);
    float scalarVal = inScalarGM.GetValue(0);

    if (static_cast<int>(scalarVal) == 1)  {
        modelCode = ONE_SCALAR_NORM_MODEL_CODE;
    }
    
    if (TILING_KEY_IS(1)) {
        if (modelCode == ONE_SCALAR_NORM_MODEL_CODE) {
            ForeachNormND<half, float, ONE_SCALAR_NORM_MODEL_CODE> op;
            op.Init(inputs, output, userWS, &tilingData);
            op.Process();
        } else {
            ForeachNormND<half, float, TWO_SCALAR_NORM_MODEL_CODE> op;
            op.Init(inputs, output, userWS, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(2)) {
        if (modelCode == ONE_SCALAR_NORM_MODEL_CODE) {
            ForeachNormND<float, float, ONE_SCALAR_NORM_MODEL_CODE> op;
            op.Init(inputs, output, userWS, &tilingData);
            op.Process();
        } else {
            ForeachNormND<float, float, TWO_SCALAR_NORM_MODEL_CODE> op;
            op.Init(inputs, output, userWS, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(4)) {
        if (modelCode == ONE_SCALAR_NORM_MODEL_CODE) {
            ForeachNormND<bfloat16_t, float, ONE_SCALAR_NORM_MODEL_CODE> op;
            op.Init(inputs, output, userWS, &tilingData);
            op.Process();
        } else {
            ForeachNormND<bfloat16_t, float, TWO_SCALAR_NORM_MODEL_CODE> op;
            op.Init(inputs, output, userWS, &tilingData);
            op.Process();
        }
    }
}
