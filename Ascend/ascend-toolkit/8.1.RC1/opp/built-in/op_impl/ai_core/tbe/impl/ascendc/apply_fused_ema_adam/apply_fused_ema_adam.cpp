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
 * \file apply_fused_ema_adam.cpp
 * \brief
 */

#include "apply_fused_ema_adam_f32.h"
#include "apply_fused_ema_adam_f_bf16.h"

using namespace FusedEmaAdam;
extern "C" __global__ __aicore__ void apply_fused_ema_adam(GM_ADDR grad, GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR s,
                                                           GM_ADDR step, GM_ADDR var_ref, GM_ADDR m_ref, GM_ADDR v_ref,
                                                           GM_ADDR s_ref, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR userWorkSpace = AscendC::GetUserWorkspace(workspace);
    TPipe pipe;
    if (TILING_KEY_IS(102)) {
        FusedEmaAdamF32<float> op;
        op.Init(grad, var, m, v, s, step, var_ref, m_ref, v_ref, s_ref, tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        FusedEmaAdamF16<half> op;
        op.Init(grad, var, m, v, s, step, var_ref, m_ref, v_ref, s_ref, tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(100)) {
        FusedEmaAdamF16<bfloat16_t> op;
        op.Init(grad, var, m, v, s, step, var_ref, m_ref, v_ref, s_ref, tiling_data, &pipe);
        op.Process();
    }
}