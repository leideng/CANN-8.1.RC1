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
 * \file quant_grouped_matmul_dequant.cpp
 * \brief
 */

#include "quant_matmul_dequant_grouped.h"

extern "C" __global__ __aicore__ void quant_grouped_matmul_dequant(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR group_list,
                                                           GM_ADDR bias, GM_ADDR x_scale, GM_ADDR x_offset, GM_ADDR smooth_scale,
                                                           GM_ADDR y, GM_ADDR usrworkspace, GM_ADDR qmmTiling) {
  SetAtomicNone();
  GET_TILING_DATA(tiling_data, qmmTiling);
  QuantMatmulDequantTilingData* __restrict tilingData = const_cast<QuantMatmulDequantTilingData *> (&tiling_data);
  if (TILING_KEY_IS(10000003)) {
    QuantMatmulDequantGrouped handle;
    handle.Init(x, quantized_weight, weight_scale, group_list, bias, x_scale, x_offset, smooth_scale, y, usrworkspace, tilingData);
    handle.Process();
  }
  SetMaskNorm();
  ResetMask();
}
