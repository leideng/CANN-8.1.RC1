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
 * \file upsample_linear_common.h
 * \brief
 */
#ifndef UPSAMPLE_LINEAR_COMMON_H
#define UPSAMPLE_LINEAR_COMMON_H
#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t BLOCK_SIZE = 32;

__aicore__ inline int64_t ROUND_UP(const int64_t x, const int64_t block_number) {
  if (block_number > 0) {
    return (x + block_number - 1) / block_number * block_number;
  }
  return 0;
}

template <typename T>
__aicore__ inline void InitGmZero(const GlobalTensor<T> &outGm, TBuf<TPosition::VECCALC> &TmpZeroTBuf, const int64_t zeroLen, const int64_t outOffset) {
  int64_t alignLen_ = BLOCK_SIZE / sizeof(T);
  LocalTensor<T> temp_zero_tensor = TmpZeroTBuf.Get<T>();
  
  Duplicate(temp_zero_tensor, (T)0.0, zeroLen);
  pipe_barrier(PIPE_ALL);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

  DataCopy(outGm[outOffset], temp_zero_tensor, ROUND_UP(zeroLen, alignLen_));
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  
  pipe_barrier(PIPE_ALL);
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilA2B(const T1 a, const T2 b) {
  if (b == 0) {
    return a;
  }
  return (a + b - 1) / b;
};

template <typename T1>
__aicore__ inline T1 weightCalculate(T1 x) {
  if (x < 0) {
    x = -1 * x;
  }
  if (x < (float)1.0) {
    return (float)1.0 - x;
  }
  return 0.0;
};

template <typename T1>
__aicore__ inline T1 Min(const T1 a, const T1 b) {
  return a < b ? a : b;
};

template <typename T1>
__aicore__ inline T1 Max(const T1 a, const T1 b) {
  return a > b ? a : b;
};

__aicore__ inline float getCenterValue(const int64_t dstIdx, const float scale, const bool align_corners) {
  if (align_corners) {
    return scale * dstIdx;
  } else {
    float rel = scale * (dstIdx + (float)0.5) - (float)0.5;
    return Max(rel, (float)0.0);
  }
};

__aicore__ inline float getLambda(const float i_rel_idx, const int64_t i_min) {
  float i_lambda = Min(Max(static_cast<float>(i_rel_idx - i_min), (float)0.0), (float)1.0);
  return i_lambda;
};

__aicore__ inline bool FloatEqual(const float a, const float b) {
  const float closeTo0 = 1e-6;
  if (a > b) {
    return a - b < closeTo0;
  } else {
    return b - a < closeTo0;
  }
};
#endif  // UPSAMPLE_LINEAR_COMMON_H