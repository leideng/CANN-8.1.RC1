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
 * \file roi_align_rotated_grad.cpp
 * \brief
 */
#include "roi_align_rotated_grad.h"

extern "C" __global__ __aicore__ void roi_align_rotated_grad(GM_ADDR grad_output, GM_ADDR rois, GM_ADDR grad_input, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetSysWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling);
    const RoiAlignRotatedGradTilingData *__restrict tilingDevice = &tilingData;
    KernelRoiAlignRotatedGrad op;
    op.Init(grad_output, rois, grad_input, tilingDevice);
    op.Process();
}