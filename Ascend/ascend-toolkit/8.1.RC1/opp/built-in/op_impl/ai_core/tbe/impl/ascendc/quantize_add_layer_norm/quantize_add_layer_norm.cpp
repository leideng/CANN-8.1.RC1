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
 * \file quantize_add_layer_norm.cpp
 * \brief
 */

#include "quantize_add_layer_norm_normal_kernel.h"
#include "quantize_add_layer_norm_single_row_kernel.h"

extern "C" __global__ __aicore__ void quantize_add_layer_norm(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
                                                              GM_ADDR bias, GM_ADDR scales, GM_ADDR zeroPoints,
                                                              GM_ADDR y, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling) {
  TPipe pipe;
  GET_TILING_DATA(tiling_data, tiling);
  GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

  // per channel
  if (TILING_KEY_IS(3000)) {
    KernelQuantizeAddLayerNormSingleRow<bfloat16_t, 3000> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(2000)) {
    KernelQuantizeAddLayerNormSingleRow<float, 2000> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(1000)) {
    KernelQuantizeAddLayerNormSingleRow<half, 1000> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(3100)) {
    KernelQuantizeAddLayerNormSingleRow<bfloat16_t, 3100> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(2100)) {
    KernelQuantizeAddLayerNormSingleRow<float, 2100> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(1100)) {
    KernelQuantizeAddLayerNormSingleRow<half, 1100> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
    op.Process();
  }

  // per channel
  if (TILING_KEY_IS(3002)) {
    KernelQuantizeAddLayerNormSingleRow<bfloat16_t, 3002> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(2002)) {
    KernelQuantizeAddLayerNormSingleRow<float, 2002> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(1002)) {
    KernelQuantizeAddLayerNormSingleRow<half, 1002> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(3102)) {
    KernelQuantizeAddLayerNormSingleRow<bfloat16_t, 3102> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(2102)) {
    KernelQuantizeAddLayerNormSingleRow<float, 2102> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(1102)) {
    KernelQuantizeAddLayerNormSingleRow<half, 1102> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
    op.Process();
  }

  // per_tensor
  else if (TILING_KEY_IS(3001)) {
    KernelQuantizeAddLayerNormNormalPerTensorKernel<bfloat16_t, float, 3001> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(2001)) {
    KernelQuantizeAddLayerNormNormalPerTensorKernel<float, float, 2001> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(1001)) {
    KernelQuantizeAddLayerNormNormalPerTensorKernel<half, float, 1001> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(3101)) {
    KernelQuantizeAddLayerNormNormalPerTensorKernel<bfloat16_t, float, 3101> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(2101)) {
    KernelQuantizeAddLayerNormNormalPerTensorKernel<float, float, 2101> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor);
    op.Process();
  } else if (TILING_KEY_IS(1101)) {
    KernelQuantizeAddLayerNormNormalPerTensorKernel<half, float, 1101> op(&pipe);
    op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor);
    op.Process();
  }
}