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
 * \file foreach_add_list_inplace.cpp
 * \brief
 */
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void foreach_add_list_inplace(GM_ADDR inputs_x, GM_ADDR inputs_y,
  GM_ADDR alpha, GM_ADDR workspace, GM_ADDR tiling) {
  // The interface has been deprecated.
  return;
}