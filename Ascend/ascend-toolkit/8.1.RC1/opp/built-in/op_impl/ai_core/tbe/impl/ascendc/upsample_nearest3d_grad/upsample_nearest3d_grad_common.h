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
 * \file upsample_nearest3d_grad_common.h
 * \brief
 */
#ifndef UPSAMPLE_NEAREST3D_GRAD_COMMON_H
#define UPSAMPLE_NEAREST3D_GRAD_COMMON_H
#include "kernel_operator.h"

using namespace AscendC;
constexpr int64_t BLOCK_SIZE = 32;

__aicore__ inline int64_t ROUND_UP(int64_t x, int64_t block_number) {
  if (block_number > 0) {
    return (x + block_number - 1) / block_number * block_number;
  }
  return 0;
}

template <typename T>
__aicore__ inline void InitGmZero(const GlobalTensor<T> &outGm, TBuf<TPosition::VECCALC> &TmpZeroTBuf, int64_t zeroLen, int64_t outOffset) {
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
#endif  // UPSAMPLE_NEAREST3D_GRAD_COMMON_H