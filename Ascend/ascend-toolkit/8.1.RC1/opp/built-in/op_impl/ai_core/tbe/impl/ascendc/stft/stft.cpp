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
 * \file stft.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "stft.h"
#include "stft_generalized.h"
#include "stft_generalized_complex.h"
#include "stft_plan_mul.h"

using namespace STFTND;
using namespace AscendC;

constexpr MatmulConfig MM_CFG = {true, false, false, 0, 0, 0, false, false, false, false, 0, 0, 0, 0, 0,
	                         0, 0, 0, true, false, false, false, false, false};
#define STFT_PERFORMANCE_IMPL(INPUT_TYPE, BUFFER_NUM)                             \
  do {                                                                                                  \
    GET_TILING_DATA_WITH_STRUCT(STFTTilingData, tiling_data_in, tiling); \
    STFTTilingData* __restrict tilingData = &tiling_data_in;            \
    StftND<INPUT_TYPE, BUFFER_NUM> op;                                          \
    op.Init(x, plan, y, userWs, tilingData);                \
    op.Process();                                                                                    \
  } while (0)

#define STFT_GENERALIZED_IMPL(INPUT_TYPE, BUFFER_NUM, MATMUL_CFG)                             \
  do {                                                                                                  \
    GET_TILING_DATA_WITH_STRUCT(STFTGeneralizedTilingData, tiling_data_in, tiling); \
    STFTGeneralizedTilingData* __restrict tilingData = &tiling_data_in;            \
    if (window != nullptr) {                                                                  \
    TPipe planPipe;                                                                           \
    StftPlanMul<float, 1> planOp;                                                                 \
    planOp.Init(plan, window, userWs, &(tilingData->planTilingData), &planPipe);                  \
    planOp.Process();                                                                             \
    planPipe.Destroy();                                                                           \
    }                                                                                             \
    TPipe pipeOp;                                                                    \
    TCubeTiling* __restrict mm0Tiling = &(tilingData->mm0TilingData);                    \
    TCubeTiling* __restrict mm1Tiling = &(tilingData->mm1TilingData);                    \
    TCubeTiling* __restrict mm2Tiling = &(tilingData->mm2TilingData);                    \
    TCubeTiling* __restrict mm3Tiling = &(tilingData->mm3TilingData);                    \
    STFTGeneralized<INPUT_TYPE, BUFFER_NUM, MATMUL_CFG> op;                                          \
    REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm0, mm0Tiling, op.mm1, mm1Tiling, op.mm2, mm2Tiling,    \
                      op.mm3, mm3Tiling);                              \
    op.Init(x, plan, window, y, userWs, tilingData, &pipeOp);                \
    op.Process();                                                                                    \
  } while (0)

#define STFT_GENERALIZED_COMPLEX_IMPL(INPUT_TYPE, BUFFER_NUM, MATMUL_CFG)                             \
  do {                                                                                                  \
    GET_TILING_DATA_WITH_STRUCT(STFTGeneralizedTilingData, tiling_data_in, tiling); \
    STFTGeneralizedTilingData* __restrict tilingData = &tiling_data_in;            \
    TPipe pipeOp;                                                                  \
    TCubeTiling* __restrict mm0Tiling = &(tilingData->mm0TilingData);                    \
    TCubeTiling* __restrict mm1Tiling = &(tilingData->mm1TilingData);                    \
    TCubeTiling* __restrict mm2Tiling = &(tilingData->mm2TilingData);                    \
    TCubeTiling* __restrict mm3Tiling = &(tilingData->mm3TilingData);                    \
    STFTGeneralizedComplex<INPUT_TYPE, BUFFER_NUM, MATMUL_CFG> op;                                          \
    REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm0, mm0Tiling, op.mm1, mm1Tiling, op.mm2, mm2Tiling,    \
                      op.mm3, mm3Tiling);                              \
    op.Init(x, plan, y, userWs, tilingData, &pipeOp);                \
    op.Process();                                                                                    \
  } while (0)

// mix算子
extern "C" __global__ __aicore__ void stft(GM_ADDR x, GM_ADDR plan, GM_ADDR window, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }

  GM_ADDR userWs = GetUserWorkspace(workspace);
  if (userWs == nullptr) {
    return;
  }

  if (TILING_KEY_IS(0)) {
    STFT_PERFORMANCE_IMPL(float, 1);
  } else if (TILING_KEY_IS(1)) {
    STFT_GENERALIZED_IMPL(float, 1, MM_CFG);
  } else if (TILING_KEY_IS(2)) {
    STFT_GENERALIZED_COMPLEX_IMPL(float, 2, MM_CFG);
  } else if (TILING_KEY_IS(3)) {
    STFT_GENERALIZED_IMPL(half, 1, MM_CFG);
  }
  return;
}
