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
 * \file chamfer_distance_grad.cpp
 * \brief
 */
#include "chamfer_distance_grad.h"

// core func
extern "C" __global__ __aicore__ void chamfer_distance_grad(GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR idx1, GM_ADDR idx2,
                                                            GM_ADDR grad_dist1, GM_ADDR grad_dist2,
                                                            GM_ADDR grad_xyz1, GM_ADDR grad_xyz2,
                                                            GM_ADDR workspace, GM_ADDR tiling_data)
{
    GET_TILING_DATA(tiling_datas, tiling_data);
    if (TILING_KEY_IS(1)) {
        ChamferDistanceGrad<float> op32(xyz1, xyz2, grad_dist1,
                                        grad_dist2, idx1, idx2,
                                        grad_xyz1, grad_xyz2,
                                        &tiling_datas);
        op32.process();
    } else if (TILING_KEY_IS(2)) {
        ChamferDistanceGrad<half> op16(xyz1, xyz2, grad_dist1,
                                       grad_dist2, idx1, idx2,
                                       grad_xyz1, grad_xyz2,
                                       &tiling_datas);
        op16.process();
    }
}