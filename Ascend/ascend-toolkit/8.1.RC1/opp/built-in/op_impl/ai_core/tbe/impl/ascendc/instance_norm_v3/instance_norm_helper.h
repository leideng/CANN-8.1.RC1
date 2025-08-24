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
 * \file instance_norm_helper.h
 * \brief
 */

#ifndef __INSTANCE_NORM_HELPER_H_
#define __INSTANCE_NORM_HELPER_H_

#include "kernel_operator.h"

using namespace AscendC;

template <typename Tp, Tp v>
struct integral_constant {
  static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

constexpr int BLOCK_SIZE = 32;

template <typename T>
__aicore__ inline void DataCopyCustomUB2GM(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                           const uint32_t count) {
  // only support count greater than 32byte
  int32_t numPerBlock = BLOCK_SIZE / sizeof(T);
  if (count % numPerBlock == 0) {
    DataCopy(dstTensor, srcTensor, count);
  } else {
    int32_t num = count / numPerBlock * numPerBlock;
    DataCopy(dstTensor, srcTensor, num);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    for (int32_t i = 0; i < numPerBlock; i++) {
      T tensorValue = srcTensor.GetValue(count - numPerBlock + i);
      srcTensor.SetValue(i, tensorValue);
    }
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    DataCopy(dstTensor[count - numPerBlock], srcTensor, numPerBlock);
  }
}

template <typename T>
__aicore__ inline void DataCopyCustomGM2UB(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor,
                                           const uint32_t count) {
  // only support count greater than 32byte
  int32_t numPerBlock = BLOCK_SIZE / sizeof(T);
  if (count % numPerBlock == 0) {
    DataCopy(dstTensor, srcTensor, count);
  } else {
    int32_t num = AlignUp(count, numPerBlock);
    DataCopy(dstTensor, srcTensor, num);
  }
}

#endif  // __INSTANCE_NORM_HELPER_H_
