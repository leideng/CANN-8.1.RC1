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
 * \file dynamic_rnn_common.h
 * \brief
 */
#ifndef _DYNAMIC_RNN_COMMON_H_
#define _DYNAMIC_RNN_COMMON_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

constexpr int64_t LSTM_GATE_SIZE = 4;
constexpr int64_t DEFAULT_QUEUE_BUFFE_SIZE = 2;
__aicore__ inline int Ceil(int a, int b) {
  if (b == 0) {
    return a;
  }
  return (a + b - 1) / b;
}

struct TRnnOffsets {
  int64_t AOffset;
  int64_t BOffset;
  int64_t COffset;
  int64_t BiasOffset;
};

struct CalcSize {
  int64_t oriBaseM;
  int64_t oriBaseN;
  int64_t tailBaseM;
  int64_t tailBaseN;
  int64_t mLoop;
  int64_t nLoop;
  int64_t hiddenMNSize;
  int64_t hiddenMKAllSize;  // mm2 左矩阵的大小
  int64_t oneBaseMTailN;
  int64_t oneLineBaseMBaseN;
  int64_t oneLineMN;
  int64_t allCellSize;
  int64_t hiddenMKSize;
  int64_t hiddenTailMKSize;
  int64_t oneTailMBaseN;
  int64_t oneTailMTailN;
  int64_t outSize;
};

// input GlobalTensors
template <typename T>
struct InputGm {
  AscendC::GlobalTensor<T> xGm;
  AscendC::GlobalTensor<T> weightGm;
  AscendC::GlobalTensor<T> biasGm;
  AscendC::GlobalTensor<T> seqLengthGm;
  AscendC::GlobalTensor<T> initHGm;
  AscendC::GlobalTensor<T> initCGm;
  AscendC::GlobalTensor<T> wciGm;
  AscendC::GlobalTensor<T> wcfGm;
  AscendC::GlobalTensor<T> wcoGm;
  AscendC::GlobalTensor<T> maskGm;
};

template <typename T>
struct OutputGm {
  __aicore__ inline OutputGm() = default;
  AscendC::GlobalTensor<T> outYGm;
  AscendC::GlobalTensor<T> outHGm;
  AscendC::GlobalTensor<T> outCGm;
  AscendC::GlobalTensor<T> outIGm;
  AscendC::GlobalTensor<T> outJGm;
  AscendC::GlobalTensor<T> outFGm;
  AscendC::GlobalTensor<T> outOGm;
  AscendC::GlobalTensor<T> outTanhCGm;
  AscendC::GlobalTensor<float> workspace;
};

struct LstmBean {
  GM_ADDR inputX;
  GM_ADDR weight;
  GM_ADDR bias;
  GM_ADDR seqLength;
  GM_ADDR initH;
  GM_ADDR initC;
  GM_ADDR wCi;
  GM_ADDR wCf;
  GM_ADDR wCo;
  GM_ADDR mask;
  GM_ADDR outputY;
  GM_ADDR outputH;
  GM_ADDR outputC;
  GM_ADDR outputI;
  GM_ADDR outputJ;
  GM_ADDR outputF;
  GM_ADDR outputO;
  GM_ADDR outputTanhC;
};

#endif
