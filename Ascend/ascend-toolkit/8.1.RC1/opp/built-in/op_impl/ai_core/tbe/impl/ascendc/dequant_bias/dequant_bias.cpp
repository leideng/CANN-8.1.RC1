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
 * \file dequant_bias.cpp
 * \brief dequant_bias kernal
 */
#include "kernel_operator.h"
#include "dequant_bias_impl.h"
#include "dequant_bias_multi.h"

using namespace AscendC;
using namespace DequantBias;

#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

extern "C" __global__ __aicore__ void dequant_bias(GM_ADDR x, GM_ADDR weight_scale, GM_ADDR activate_scale,
                                                   GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(10100)) {
        if (tilingData.N <= 8192) { 
            DequantBias::DequantBiasImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, DTYPE_BIAS, DTYPE_Y, false> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        } else { DequantBias::DequantBiasMultiImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, DTYPE_BIAS, DTYPE_Y, false> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(10110)) {
        if (tilingData.N <= 8192) { 
            DequantBias::DequantBiasImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, int32_t, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        } else { DequantBias::DequantBiasMultiImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, int32_t, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(10111)) {
        if (tilingData.N <= 8192) { 
            DequantBias::DequantBiasImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, float, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        } else { DequantBias::DequantBiasMultiImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, float, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(10112)) {
        if (tilingData.N <= 8192) { 
            DequantBias::DequantBiasImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, half, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        } else { DequantBias::DequantBiasMultiImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, half, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(10113)) {
        if (tilingData.N <= 8192) { 
            DequantBias::DequantBiasImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, bfloat16_t, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        } else { DequantBias::DequantBiasMultiImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, bfloat16_t, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        }
    }
}