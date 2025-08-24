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
 * \file common.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_COMMON_H
#define MOE_GATING_TOP_K_COMMON_H

#include "kernel_operator.h"

namespace MoeGatingTopK {
using namespace AscendC;
const float MIN_FP32 = *(float *)(&F32_NEG_INF);
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t BLOCK_BYTES = 32;

constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;

constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;


__aicore__ inline int64_t Ceil(int64_t a, int64_t b) {
  if (b == 0) {
    return 0;
  }
  return (a + b - 1) / b;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes) {
  if (bytes == 0) {
    return 0;
  }
  return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes) {
  return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <typename T>
__aicore__ inline T Min(T a, T b) {
  return a > b ? b : a;
}

template <typename T>
__aicore__ inline T Max(T a, T b) {
  return a < b ? b : a;
}

}  // namespace MoeGatingTopK
#endif  // MOE_GATING_TOP_K_COMMON_H