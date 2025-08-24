/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file sync.h
 * \brief
 */
#ifndef ATP_SYNC_H_
#define ATP_SYNC_H_

#include "kernel_operator.h"

/**
  * 调用get_buf函数
  * @tparam p TPosition的类型
  * @param bufID 当前处理的buf的真是ID
  */
template <TPosition p>
__aicore__ inline void GetBuf(uint8_t bufID) {
  if constexpr (p == TPosition::VECIN) {
    get_buf(PIPE_MTE2, bufID, 0);
  } else if constexpr (p == TPosition::VECCALC) {
    get_buf(PIPE_V, bufID, 0);
  } else if constexpr (p == TPosition::VECOUT) {
    get_buf(PIPE_MTE3, bufID, 0);
  }
}

/**
  * 调用rls_buf函数
  * @tparam p TPosition的类型
  * @param bufID 当前处理的buf的真实ID
  */
template <TPosition p>
__aicore__ inline void RlsBuf(uint8_t bufID) {
  if constexpr (p == TPosition::VECIN) {
    rls_buf(PIPE_MTE2, bufID, 0);
  } else if constexpr (p == TPosition::VECCALC) {
    rls_buf(PIPE_V, bufID, 0);
  } else if constexpr (p == TPosition::VECOUT) {
    rls_buf(PIPE_MTE3, bufID, 0);
  }
}

/**
  * 插入GetBuf同步函数
  * @tparam p TPosition的类型
  * @param bufID 当前处理的buf的真实ID
  */
template <TPosition p>
__aicore__ void inline GetTensor(uint8_t bufID) {
  RUN_LOG("GetTensor: ID: %d, Position = %d", bufID, p);
  GetBuf<p>(bufID);
}

/**
  * 插入RlsBuf同步函数
  * @tparam p TPosition的类型
  * @param bufID 当前处理的buf的真实ID
  */
template <TPosition p>
__aicore__ void inline ReleaseTensor(uint8_t bufID) {
  RUN_LOG("ReleaseTensor: ID: %d, Position = %d", bufID, p);
  RlsBuf<p>(bufID);
}

#endif  // ATP_SYNC_H_