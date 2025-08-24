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
 * \file conv2d_transpose_v2.cpp
 * \brief
 */
#include "../conv3d_backprop_input_v2/conv3d_backprop_input_v2.h"
#include "../conv3d_backprop_input_v2/conv3d_backprop_input_v2_init_output.h"

using namespace AscendC;
#ifndef Y_FORMAT_3D
#if defined(FORMAT_Y) && FORMAT_Y == FORMAT_NC1HWC0
#define Y_FORMAT_3D FORMAT_NDC1HWC0
#else
#define Y_FORMAT_3D FORMAT_NCDHW
#endif
#endif

extern "C" __global__ __aicore__ void conv2d_transpose_v2(GM_ADDR input_size, GM_ADDR x, GM_ADDR filter, GM_ADDR bias,
                                                          GM_ADDR offset_w, GM_ADDR y, GM_ADDR workSpace,
                                                          GM_ADDR tiling)
{
    if (workSpace == nullptr) {
        return;
    }

    GM_ADDR usrWsp = GetUserWorkspace(workSpace);
    if (usrWsp == nullptr) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(Conv3DBackpropInputV2TilingData, tilingData, tiling);

#if __CCE_AICORE__ == 220
    if constexpr (Y_FORMAT_3D == FORMAT_NCDHW) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    } else if (tilingData.conv3DDxTiling.initOutputFlag == static_cast<int32_t>(InitOutputFlag::L0_INIT)) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    }
#endif

    if (tilingData.conv3DDxTiling.initOutputFlag == static_cast<int32_t>(InitOutputFlag::L0_INIT)) {
        // init output with L0C now
        Conv3dDxInitOutput<DTYPE_Y, Y_FORMAT_3D, InitOutputFlag::L0_INIT> opInitOutput;
        opInitOutput.Init(y, &tilingData);
        opInitOutput.Process();
        opInitOutput.Destroy();
    }

    if (TILING_KEY_IS(0)) {
        Conv3dDx<DTYPE_FILTER, FORMAT_FILTER, DTYPE_X, FORMAT_X, DTYPE_Y, Y_FORMAT_3D,
                 Convolution3DBackprop::B2Condition::BASEK_LT_HKWK>
            op;
        op.Init(filter, x, y, usrWsp, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        Conv3dDx<DTYPE_FILTER, FORMAT_FILTER, DTYPE_X, FORMAT_X, DTYPE_Y, Y_FORMAT_3D,
                 Convolution3DBackprop::B2Condition::BASEK_GE_HKWK>
            op;
        op.Init(filter, x, y, usrWsp, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        Conv3dDx<DTYPE_FILTER, FORMAT_FILTER, DTYPE_X, FORMAT_X, DTYPE_Y, Y_FORMAT_3D,
                 Convolution3DBackprop::B2Condition::HKWK_EQ_ONE>
            op;
        op.Init(filter, x, y, usrWsp, &tilingData);
        op.Process();
    } if (TILING_KEY_IS(10)) { // conv3d_transpose_videogpt_f240_h256_net_ID_2 and conv3d_transpose_videogpt_f240_h256_net_ID_1
        Conv3dDx<DTYPE_FILTER, FORMAT_FILTER, DTYPE_X, FORMAT_X, DTYPE_Y, Y_FORMAT_3D,
                 Convolution3DBackprop::B2Condition::BASEK_LT_HKWK, true>
            op;
        op.Init(filter, x, y, usrWsp, &tilingData);
        op.Process();
    }
}