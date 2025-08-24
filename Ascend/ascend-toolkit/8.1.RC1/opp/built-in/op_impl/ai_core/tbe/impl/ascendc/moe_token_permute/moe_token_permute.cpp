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
 * \file moe_token_permute.cpp
 * \brief
 */
#include "moe_mrgsort_out.h"
#include "moe_mrgsort.h"
#include "moe_sort_multi_core.h"
#include "moe_sort_one_core.h"
#include "moe_index_copy.h"
#include "moe_index_copy_spilt_d.h"
#if !defined(DTYPE_TOKENS)
#define DTYPE_TOKENS bfloat16_t 
#endif
#if !defined(DTYPE_INDICES)
#define DTYPE_INDICES int64_t 
#endif
using namespace AscendC;
using namespace MoeTokenPermute;
#define GENERAL_OP_IMPL(sortClass1, sortClass2, indexCopyClass, ...)                                                               \
    do {                                                                                                                \
        sortClass1<DTYPE_INDICES> op;                                                                                  \
        op.Init(indices, sortedIndices, userWS, t, &sortPipe);                                                                                   \
        op.Process();                                                                                                   \
        sortPipe.Destroy();                               \
        sortClass2<int32_t> op2;                               \
        op2.Init(sortedIndices, sortedIndices, userWS, t, &sortPipe2);                               \
        op2.Process();                               \
        sortPipe2.Destroy();                               \
        indexCopyClass<__VA_ARGS__> indexCopyOp;                               \
        indexCopyOp.Init(tokens, sortedIndices, permuteTokens, t, &MoeindexCopyPipe);                               \
        indexCopyOp.Process();                               \
    } while (0)

extern "C" __global__ __aicore__ void moe_token_permute(GM_ADDR tokens, GM_ADDR indices, GM_ADDR permuteTokens,
                                                        GM_ADDR sortedIndices,
                                                        GM_ADDR workspace, GM_ADDR tiling) {
  if (g_coreType == AIC) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);
  if (workspace == nullptr) {
    return;
  }

  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  auto t = &tilingData;
  TPipe sortPipe;
  TPipe sortPipe2;
  TPipe MoeindexCopyPipe;
  if (TILING_KEY_IS(1)) {
    GENERAL_OP_IMPL(MoeSortOneCore, MoeSortOneCore, MoeindexCopyOp, DTYPE_TOKENS, false);
  } else if (TILING_KEY_IS(3)) {
    GENERAL_OP_IMPL(MoeSortOneCore, MoeSortOneCore, MoeindexCopySpiltDOp, DTYPE_TOKENS, false);
  } else if (TILING_KEY_IS(2)) {
    GENERAL_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopyOp, DTYPE_TOKENS, false);
  } else if (TILING_KEY_IS(4)) {
    GENERAL_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopySpiltDOp, DTYPE_TOKENS, false);
  } else if (TILING_KEY_IS(5)) {
    GENERAL_OP_IMPL(MoeSortOneCore, MoeSortOneCore, MoeindexCopyOp, DTYPE_TOKENS, true);
  } else if (TILING_KEY_IS(7)) {
    GENERAL_OP_IMPL(MoeSortOneCore, MoeSortOneCore, MoeindexCopySpiltDOp, DTYPE_TOKENS, true);
  } else if (TILING_KEY_IS(6)) {
    GENERAL_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopyOp, DTYPE_TOKENS, true);
  } else if (TILING_KEY_IS(8)) {
    GENERAL_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopySpiltDOp, DTYPE_TOKENS, true);
  }
}
