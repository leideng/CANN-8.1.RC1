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
 * \file multi_scale_deformable_attention_grad.cpp
 * \brief
 */
#include "multi_scale_deformable_attention_grad.h"

using namespace AscendC;
// core func
extern "C" __global__ __aicore__ void multi_scale_deformable_attention_grad(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm,
                                                                            GM_ADDR level_start_index_gm, GM_ADDR sampling_loc_gm,
                                                                            GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm,
                                                                            GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm,
                                                                            GM_ADDR grad_attn_weight_gm, GM_ADDR workspace,
                                                                            GM_ADDR tiling_data)
{
    
    TPipe pipe;
    GET_TILING_DATA(tiling_datas, tiling_data);
    MultiScaleDeformableAttentionGrad op;
    op.Init(value_gm, spatial_shapes_gm, level_start_index_gm, sampling_loc_gm, attn_weight_gm, grad_output_gm,
            grad_value_gm, grad_sampling_loc_gm, grad_attn_weight_gm, &tiling_datas, &pipe);
    op.InitBuffer();
    op.GetLocalTensor();
    op.ClearOutput();
    op.Process();
    op.ReleaseEventID();
}
