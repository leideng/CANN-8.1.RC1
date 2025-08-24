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
 * \file foreach_non_finite_check_and_unscale.cpp
 * \brief
 */
#include "foreach_non_finite_check_and_unscale_n_d.h"

using namespace ForeachNonFiniteCheckAndUnscale;

extern "C" __global__ __aicore__ void foreach_non_finite_check_and_unscale(GM_ADDR scaled_grads, GM_ADDR found_inf,
                                                                           GM_ADDR inv_scale, GM_ADDR workspace,
                                                                           GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    if (TILING_KEY_IS(1)) {
        ForeachNonFiniteCheckAndUnscaleND<float> op;
        op.Init(scaled_grads, found_inf, inv_scale, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachNonFiniteCheckAndUnscaleND<half> op;
        op.Init(scaled_grads, found_inf, inv_scale, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
#if (__CCE_AICORE__ > 200)
        ForeachNonFiniteCheckAndUnscaleND<bfloat16_t> op;
        op.Init(scaled_grads, found_inf, inv_scale, userWS, &tilingData);
        op.Process();
#endif
    }
}