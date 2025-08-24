/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
 * \file adaptive_avg_pool3d.cpp
 * \brief
 */

#include "kernel_operator.h"

#include "adaptive_avg_pool3d_split_c.h"
#include "adaptive_avg_pool3d_multi_w.h"
#include "adaptive_avg_pool3d_split_w.h"

using namespace AdaptiveAvgPool3d;

#define DISPATCH_OP_IMPL(KernelImpl, ...)                                                                              \
  do {                                                                                                                 \
    KernelImpl<__VA_ARGS__> op;                                                                                        \
    TPipe tPipe;                                                                                                       \
    op.Init(x, y, workspace, &tilingData, &tPipe);                                                                     \
    op.Process();                                                                                                      \
  } while (0)


extern "C" __global__ __aicore__ void adaptive_avg_pool3d(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  if (TILING_KEY_IS(11)) {
    DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitC, half, 1);
#if __CCE_AICORE__ >= 220
  } else if (TILING_KEY_IS(10)) {
    DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitC, bfloat16_t, 1);
#endif
  } else if (TILING_KEY_IS(12)) {
    DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitC, float, 1);
#if __CCE_AICORE__ >= 220
  } else if (TILING_KEY_IS(20)) {
    DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitW, bfloat16_t, 1);
#endif
  } else if (TILING_KEY_IS(21)) {
    DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitW, half, 1);
  } else if (TILING_KEY_IS(22)) {
    DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitW, float, 1);
#if __CCE_AICORE__ >=220
  } else if (TILING_KEY_IS(30)) {
    DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dMultiW, bfloat16_t, 1);
#endif
  } else if (TILING_KEY_IS(31)) {
    DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dMultiW, half, 1);
  } else if (TILING_KEY_IS(32)) {
    DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dMultiW, float, 1);
  }
}
