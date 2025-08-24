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
 * \file resize_upsample_trilinear.cpp
 * \brief
 */
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 300
#include "resize_upsample_trilinear_310p.h"
#else
#include "resize_upsample_trilinear.h"
#endif

using namespace UpsampleTrilinearNs;

extern "C" __global__ __aicore__ void resize_upsample_trilinear(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                                GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  const UpsampleTrilinearTilingData* __restrict tilingData = &tiling_data;
  
  GM_ADDR userWs = GetUserWorkspace(workspace);

#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 300
  #define INIT_310P_PROCESS                                                                                    \
    op.Init(input, output, userWs, &tiling_data);                                                             \
    op.Process()
  int64_t outw = tilingData->output_w;
  int64_t outh = tilingData->output_h;
  int64_t outd = tilingData->output_d;
  if (TILING_KEY_IS(1000)) {
    KernelUpsampleTrilinear310p<half> op;
    INIT_310P_PROCESS;
  } else if (TILING_KEY_IS(3000)) {
    KernelUpsampleTrilinear310p<float> op;
    INIT_310P_PROCESS;
  }
#else
  const TCubeTiling* __restrict matmulTilingW = &(tilingData->matmul_tiling_w);
  const TCubeTiling* __restrict matmulTilingH = &(tilingData->matmul_tiling_h);
  const TCubeTiling* __restrict matmulTilingD = &(tilingData->matmul_tiling_d);
  #define INIT_AND_PROCESS                                                                                    \
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmul_w, matmulTilingW, op.matmul_h, matmulTilingH, \
                      op.matmul_d, matmulTilingD);                                                            \
    op.Init(input, output, userWs, &tiling_data);                                                             \
    op.Process()

    if (TILING_KEY_IS(1000)) {
      KernelUpsampleTrilinear<half> op;
      INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2000)) {
      KernelUpsampleTrilinear<bfloat16_t> op;
      INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3000)) {
      KernelUpsampleTrilinear<float> op;
      INIT_AND_PROCESS;
    }
#endif
}