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
 * \file ascend_quant_v2.cpp
 * \brief
 */

#include "ascend_quant_v2_fp16.h"
#include "ascend_quant_v2_fp32.h"
#include "ascend_quant_v2_bf16.h"

#define KEY_PER_CHANNEL 0
#define KEY_PER_TENSOR 1
#define KEY_PER_HEAD 2

using namespace AscendC;

extern "C" __global__ __aicore__ void ascend_quant_v2(GM_ADDR x, GM_ADDR scale, GM_ADDR offset,
                                                      GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);

  if constexpr (std::is_same<DTYPE_X, half>::value) {
    if (TILING_KEY_IS(KEY_PER_CHANNEL)) {
      AscendQuantV2::AscendQuantV2PerChannelFP16<DTYPE_X> op;
      op.Init(x, scale, offset, y, &tilingData);
      op.Process();
    } else if (TILING_KEY_IS(KEY_PER_TENSOR)) {
      AscendQuantV2::AscendQuantV2PerTensorFP16<DTYPE_X> op;
      op.Init(x, scale, offset, y, &tilingData);
      op.Process();
    }
  } else if constexpr (std::is_same<DTYPE_X, float>::value) {
    if (TILING_KEY_IS(KEY_PER_CHANNEL)) {
      AscendQuantV2::AscendQuantV2PerChannelFP32<DTYPE_X> op;
      op.Init(x, scale, offset, y, &tilingData);
      op.Process();
    } else if (TILING_KEY_IS(KEY_PER_TENSOR)) {
      AscendQuantV2::AscendQuantV2PerTensorFP32<DTYPE_X> op;
      op.Init(x, scale, offset, y, &tilingData);
      op.Process();
    }
#if __CCE_AICORE__ == 220
  } else if constexpr (std::is_same<DTYPE_X, bfloat16_t>::value) {
    if (TILING_KEY_IS(KEY_PER_CHANNEL)) {
      AscendQuantV2::AscendQuantV2PerChannnelBF16<DTYPE_X> op;
      op.Init(x, scale, offset, y, &tilingData);
      op.Process();
    } else if (TILING_KEY_IS(KEY_PER_TENSOR)) {
      AscendQuantV2::AscendQuantV2PerTensorBF16<DTYPE_X> op;
      op.Init(x, scale, offset, y, &tilingData);
      op.Process();
    }
  }
  if (TILING_KEY_IS(KEY_PER_HEAD)) {
    AscendQuantV2::AscendQuantV2PerHead<DTYPE_X> op;
    op.Init(x, scale, offset, y, &tilingData);
    op.Process();
#endif
  }
}