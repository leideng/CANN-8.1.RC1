/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file trans_quant_param_v2.cpp
 * \brief
 */
#include "trans_quant_param_v2.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void trans_quant_param_v2(GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace,
                                                           GM_ADDR tiling) {
  if (workspace == nullptr) {return;}
  GM_ADDR user1 = GetUserWorkspace(workspace);
  if (user1 == nullptr) {return;}
  GET_TILING_DATA(tilingData, tiling);
  TPipe tPipe;
  TransQuantParamV2 op;
  op.Init(scale, offset, y, user1, &tilingData, &tPipe);
  op.Process();
}