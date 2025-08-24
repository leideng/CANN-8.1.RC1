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
 * \file apply_adam_w_v2.cpp
 * \brief
 */

#include "apply_adam_w_v2_fp.h"
#include "apply_adam_w_v2_b16.h"
#include "apply_adam_w_v2_mix_dtype.h"

using namespace ApplyAdamWV2;
extern "C" __global__ __aicore__ void apply_adam_w_v2(GM_ADDR var, GM_ADDR expAvg, GM_ADDR expAvgSq,
    GM_ADDR grad, GM_ADDR step, GM_ADDR maxGradNorm, GM_ADDR workspace, GM_ADDR tiling) {
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(101)) {
        ApplyAdamWV2B16<bfloat16_t, float> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(102)) {
        ApplyAdamWV2B16<bfloat16_t, int64_t> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(103)) {
        ApplyAdamWV2B16<half, float> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    }  else if (TILING_KEY_IS(104)) {
        ApplyAdamWV2B16<half, int64_t> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(105)) {
        ApplyAdamWV2Fp<float, float> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(106)) {
        ApplyAdamWV2Fp<float, int64_t> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    }  else if (TILING_KEY_IS(107)) {
        ApplyAdamWV2MixType<float, half, float> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    }  else if (TILING_KEY_IS(108)) {
        ApplyAdamWV2MixType<float, half, int64_t> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(109)) {
        ApplyAdamWV2MixType<float, bfloat16_t, float> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    }  else if (TILING_KEY_IS(110)) {
        ApplyAdamWV2MixType<float, bfloat16_t, int64_t> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    }
}
