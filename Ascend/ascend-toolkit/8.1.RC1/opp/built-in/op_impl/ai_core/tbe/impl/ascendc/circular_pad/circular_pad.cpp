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
 * \file circular_pad.cpp
 * \brief
 */
#include "circular_pad_2d.h"
#include "circular_pad_3d.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void circular_pad(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (TILING_KEY_IS(211)) {
        CircularPad2D<int8_t> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(212)) {
        CircularPad2D<int8_t> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(221)) {
        CircularPad2D<half> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(222)) {
        CircularPad2D<half> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(231)) {
        CircularPad2D<bfloat16_t> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(232)) {
        CircularPad2D<bfloat16_t> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(241)) {
        CircularPad2D<float> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(242)) {
        CircularPad2D<float> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(251)) {
        CircularPad2D<int32_t> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(252)) {
        CircularPad2D<int32_t> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }

    else if (TILING_KEY_IS(311)) {
        CircularPad3D<int8_t> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(312)) {
        CircularPad3D<int8_t> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(321)) {
        CircularPad3D<half> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(322)) {
        CircularPad3D<half> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(331)) {
        CircularPad3D<bfloat16_t> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(332)) {
        CircularPad3D<bfloat16_t> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(341)) {
        CircularPad3D<float> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(342)) {
        CircularPad3D<float> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(351)) {
        CircularPad3D<int32_t> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(352)) {
        CircularPad3D<int32_t> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    } else {
        return;
    }
}