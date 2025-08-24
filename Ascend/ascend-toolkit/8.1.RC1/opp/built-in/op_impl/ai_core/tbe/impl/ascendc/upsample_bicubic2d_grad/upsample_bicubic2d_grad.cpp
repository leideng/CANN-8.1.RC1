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
 * \file upsample_bicubic2d_grad.cpp
 * \brief
 */
#include "upsample_bicubic2d_grad_base.cpp"
#include "upsample_bicubic2d_grad_dc.h"

extern "C" __global__ __aicore__ void upsample_bicubic2d_grad(GM_ADDR grad_output, GM_ADDR grad_input, GM_ADDR workspace, GM_ADDR tiling_addr) {
  set_mask_norm();
  GET_TILING_DATA(tiling_data, tiling_addr);
  const UpsampleBicubic2dGradTilingData* __restrict tilingData = &tiling_data;
  const TCubeTiling* __restrict MMHTiling = &(tilingData->MMParamH);
  const TCubeTiling* __restrict MMWTiling = &(tilingData->MMParamW);
  if (TILING_KEY_IS(10000001)) {
    if(tilingData->dataType == 0) {
      UpsampleBicubic2dGradBase<float> opHandle;
      REGIST_MATMUL_OBJ(&opHandle.pipe, GetSysWorkSpacePtr(), opHandle.MMH, MMHTiling, opHandle.MMW, MMWTiling);
      opHandle.Init(grad_output, grad_input, workspace, &tiling_data);
      opHandle.Process();
    } else if(tilingData->dataType == 1) {
      UpsampleBicubic2dGradBase<half> opHandle;
      REGIST_MATMUL_OBJ(&opHandle.pipe, GetSysWorkSpacePtr(), opHandle.MMH, MMHTiling, opHandle.MMW, MMWTiling);
      opHandle.Init(grad_output, grad_input, workspace, &tiling_data);
      opHandle.Process();
    } else if(tilingData->dataType == 27) {
      UpsampleBicubic2dGradBase<bfloat16_t> opHandle;
      REGIST_MATMUL_OBJ(&opHandle.pipe, GetSysWorkSpacePtr(), opHandle.MMH, MMHTiling, opHandle.MMW, MMWTiling);
      opHandle.Init(grad_output, grad_input, workspace, &tiling_data);
      opHandle.Process();
    }
  } else if(TILING_KEY_IS(10000002)) {
    if(tilingData->dataType == 0) {
      UpsampleBicubic2dGrad::UpsampleBicubic2dGradDCND<float> op;
      REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, MMWTiling, op.matmulH, MMHTiling);
      op.Init(grad_output, grad_input, workspace, &tiling_data);
      op.Process();
    } else if(tilingData->dataType == 1) {
      UpsampleBicubic2dGrad::UpsampleBicubic2dGradDCND<half> op;
      REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, MMWTiling, op.matmulH, MMHTiling);
      op.Init(grad_output, grad_input, workspace, &tiling_data);
      op.Process();
    } else if(tilingData->dataType == 27) {
      UpsampleBicubic2dGrad::UpsampleBicubic2dGradDCND<bfloat16_t> op;
      REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulW, MMWTiling, op.matmulH, MMHTiling);
      op.Init(grad_output, grad_input, workspace, &tiling_data);
      op.Process();
    }
  }
}
