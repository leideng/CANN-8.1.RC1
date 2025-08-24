/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file moe_tutel_combine_x.cpp
 * \brief
 */
#include "moe_tutel_combine_x_32b.h"
#include "moe_tutel_combine_x_16b.h"

extern "C" __global__ __aicore__ void moe_tutel_combine_x(GM_ADDR y_grad, GM_ADDR gates, GM_ADDR indices,
                                                          GM_ADDR locations, GM_ADDR x_grad, GM_ADDR workspace,
                                                          GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  if (TILING_KEY_IS(1)) {
    MoeTutelCombineXFloat<float> op;
    op.Init(y_grad, gates, indices, locations, x_grad, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(2)) {
    MoeTutelCombineXFloat<half> op;
    op.Init(y_grad, gates, indices, locations, x_grad, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(0)) {
    MoeTutelCombineX<bfloat16_t> op;
    op.Init(y_grad, gates, indices, locations, x_grad, &tilingData);
    op.Process();
  }
}