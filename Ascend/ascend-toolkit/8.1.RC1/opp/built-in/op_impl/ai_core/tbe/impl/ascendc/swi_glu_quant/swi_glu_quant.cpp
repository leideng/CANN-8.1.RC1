/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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

/* !
 * \file swi_glu_quant.cpp
 * \brief
 */

#include "swi_glu_quant.h"
#include "swi_glu_quant_static.h"
using namespace AscendC;
using namespace SwiGluQuantOpt;

extern "C" __global__ __aicore__ void swi_glu_quant(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR offsets,
                                                    GM_ADDR group_index, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace,
                                                    GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }
  GET_TILING_DATA(tiling_data, tiling);
  if (GetBlockIdx() >= tiling_data.realCoreNum) {
    return;
  }
  TPipe pipe;

  #if ORIG_DTYPE_X == DT_BF16
     if (TILING_KEY_IS(206)) {
      SwiGluQuant<bfloat16_t, int8_t> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(204)) {
      SwiGluQuantStatic<bfloat16_t, int8_t, QuantType::STATIC_PER_TENSOR> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(205)) {
      SwiGluQuantStatic<bfloat16_t, int8_t, QuantType::STATIC_PER_CHANNEL> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    }
  #elif ORIG_DTYPE_X == DT_FLOAT16
    if (TILING_KEY_IS(106)) {
      SwiGluQuant<half, int8_t> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    }
    else if (TILING_KEY_IS(104)) {
      SwiGluQuantStatic<half, int8_t, QuantType::STATIC_PER_TENSOR> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(105)) {
      SwiGluQuantStatic<half, int8_t, QuantType::STATIC_PER_CHANNEL> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    }
  #elif ORIG_DTYPE_X == DT_FLOAT
    if (TILING_KEY_IS(306)) {
      SwiGluQuant<float, int8_t> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(304)) {
      SwiGluQuantStatic<float, int8_t, QuantType::STATIC_PER_TENSOR> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(305)) {
      SwiGluQuantStatic<float, int8_t, QuantType::STATIC_PER_CHANNEL> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    }
  #endif
}