/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file foreach_functors.h
 * \brief
 */
#ifndef __FOR_EACH_FUNCTORS_H__
#define __FOR_EACH_FUNCTORS_H__

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_LENGTH_BYTE = 32;
constexpr int32_t FULL_TENSOR_NUM = 8;
constexpr int32_t MAX_INPUT_TENSOR_NUM = 4;
namespace foreachFunctors {
using namespace AscendC;

// util of function
template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b) {
  return (a + b - 1) / b * b;
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b) {
  return (a + b - 1) / b;
};

typedef struct {
  int32_t tensorIdx = 0;
  int64_t offset = 0;
  int64_t dataCount = 0;
} TensorSlice_t;

// Unary Functors
template <typename T, typename scalarDataType, int depth, int rdDepth>
struct UnaryOpListFunctor {
  template <typename Op>
  __aicore__ __inline__ void operator()(TQue<QuePosition::VECIN, BUFFER_NUM> (&dataInQueue)[MAX_INPUT_TENSOR_NUM],
                                        TQue<QuePosition::VECOUT, BUFFER_NUM>& dataOutQueue, uint32_t dataCount,
                                        Op op) {
    LocalTensor<T> rDataLocal[rdDepth];
    LocalTensor<T> wDataLocal;
    for (int i = 0; i < rdDepth; i++) {
      rDataLocal[i] = dataInQueue[i].DeQue<T>();
    }
    wDataLocal = dataOutQueue.AllocTensor<T>();
    op(wDataLocal, rDataLocal[0], dataCount);
    dataOutQueue.EnQue(wDataLocal);
    for (int i = 0; i < rdDepth; i++) {
      dataInQueue[i].FreeTensor(rDataLocal[i]);
    }
  }
};

template <typename T, typename scalarDataType, int depth, int rdDepth>
struct UnaryOpScalarFunctor {
  template <typename Op>
  __aicore__ __inline__ void operator()(TQue<QuePosition::VECIN, BUFFER_NUM> (&dataInQueue)[MAX_INPUT_TENSOR_NUM],
                                        TQue<QuePosition::VECOUT, BUFFER_NUM>& dataOutQueue, uint32_t dataCount, Op op,
                                        GM_ADDR scalar) {
    LocalTensor<T> rDataLocal[rdDepth];
    LocalTensor<T> wDataLocal;
    for (int i = 0; i < rdDepth; i++) {
      rDataLocal[i] = dataInQueue[i].DeQue<T>();
    }
    wDataLocal = dataOutQueue.AllocTensor<T>();
    GlobalTensor<scalarDataType> gScalar;
    gScalar.SetGlobalBuffer((__gm__ scalarDataType*)scalar, 1);
    op(wDataLocal, rDataLocal[0], scalarDataType(gScalar.GetValue(0)), dataCount);
    dataOutQueue.EnQue(wDataLocal);
    for (int i = 0; i < rdDepth; i++) {
      dataInQueue[i].FreeTensor(rDataLocal[i]);
    }
  }
};

template <typename T, typename scalarDataType, int depth, int rdDepth>
struct UnaryOpScalarListFunctor {
  template <typename Op>
  __aicore__ __inline__ void operator()(TQue<QuePosition::VECIN, BUFFER_NUM> (&dataInQueue)[MAX_INPUT_TENSOR_NUM],
                                        TQue<QuePosition::VECOUT, BUFFER_NUM>& dataOutQueue, uint32_t sliceCount,
                                        TensorSlice_t (&tensorSlice)[FULL_TENSOR_NUM], Op op, GM_ADDR scalarList) {
    LocalTensor<T> rDataLocal[rdDepth];
    LocalTensor<T> wDataLocal;
    for (int i = 0; i < rdDepth; i++) {
      rDataLocal[i] = dataInQueue[i].DeQue<T>();
    }
    wDataLocal = dataOutQueue.AllocTensor<T>();
    uint32_t offset = 0;
    int32_t perBlockCount = BLOCK_LENGTH_BYTE / sizeof(T);
    for (uint32_t k = 0; k < sliceCount; k++) {
      GlobalTensor<scalarDataType> scalar;
      scalar.SetGlobalBuffer(((__gm__ scalarDataType*)(scalarList) + tensorSlice[k].tensorIdx), 1);
      uint32_t dataCount = CeilAlign(tensorSlice[k].dataCount, perBlockCount);
      op(wDataLocal[offset], rDataLocal[0][offset], scalarDataType(scalar.GetValue(0)), dataCount);
      offset += dataCount;
    }
    dataOutQueue.EnQue(wDataLocal);
    for (int i = 0; i < rdDepth; i++) {
      dataInQueue[i].FreeTensor(rDataLocal[i]);
    }
  }
};

// Binary Functors
template <typename T, int depth, int rdDepth>
struct BinaryOpListFunctor {
  template <typename Op>
  __aicore__ __inline__ void operator()(TQue<QuePosition::VECIN, BUFFER_NUM> (&dataInQueue)[MAX_INPUT_TENSOR_NUM],
                                        TQue<QuePosition::VECOUT, BUFFER_NUM>& dataOutQueue, uint32_t ubFactorElement,
                                        Op op) {
    LocalTensor<T> rDataLocal[rdDepth];
    LocalTensor<T> wDataLocal;
    for (int i = 0; i < rdDepth; i++) {
      rDataLocal[i] = dataInQueue[i].DeQue<T>();
    }
    wDataLocal = dataOutQueue.AllocTensor<T>();
    op(wDataLocal, rDataLocal[0], rDataLocal[1], ubFactorElement);
    dataOutQueue.EnQue(wDataLocal);
    for (int i = 0; i < rdDepth; i++) {
      dataInQueue[i].FreeTensor(rDataLocal[i]);
    }
  }
};

template <typename T, typename scalarDataType, int depth, int rdDepth>
struct BinaryOpScalarFunctor {
  template <typename Op>
  __aicore__ __inline__ void operator()(TQue<QuePosition::VECIN, BUFFER_NUM> (&dataInQueue)[MAX_INPUT_TENSOR_NUM],
                                        TQue<QuePosition::VECOUT, BUFFER_NUM>& dataOutQueue, uint32_t dataCount, Op op,
                                        GM_ADDR scalar) {
    LocalTensor<T> rDataLocal[rdDepth];
    LocalTensor<T> wDataLocal;
    for (int i = 0; i < rdDepth; i++) {
      rDataLocal[i] = dataInQueue[i].DeQue<T>();
    }
    wDataLocal = dataOutQueue.AllocTensor<T>();
    GlobalTensor<scalarDataType> scalarLocal;
    scalarLocal.SetGlobalBuffer((__gm__ scalarDataType*)(scalar), 1);
    op(wDataLocal, rDataLocal[0], rDataLocal[1], scalarDataType(scalarLocal.GetValue(0)), dataCount);
    dataOutQueue.EnQue(wDataLocal);
    for (int i = 0; i < rdDepth; i++) {
      dataInQueue[i].FreeTensor(rDataLocal[i]);
    }
  }
};

template <typename T, typename scalarDataType, const int depth, const int rdDepth>
struct BinaryOpScalarListFunctor {
  template <typename Op>
  __aicore__ __inline__ void operator()(TQue<QuePosition::VECIN, BUFFER_NUM> (&dataInQueue)[MAX_INPUT_TENSOR_NUM],
                                        TQue<QuePosition::VECOUT, BUFFER_NUM>& dataOutQueue, uint32_t sliceCount,
                                        TensorSlice_t (&tensorSlice)[FULL_TENSOR_NUM], Op op, GM_ADDR scalarList) {
    LocalTensor<T> rDataLocal[rdDepth];
    LocalTensor<T> wDataLocal;
    for (int i = 0; i < rdDepth; i++) {
      rDataLocal[i] = dataInQueue[i].DeQue<T>();
    }
    wDataLocal = dataOutQueue.AllocTensor<T>();
    uint32_t offset = 0;
    int32_t perBlockCount = BLOCK_LENGTH_BYTE / sizeof(T);
    for (uint32_t k = 0; k < sliceCount; k++) {
      GlobalTensor<scalarDataType> scalar;
      scalar.SetGlobalBuffer(((__gm__ scalarDataType*)(scalarList) + tensorSlice[k].tensorIdx), 1);
      uint32_t dataCount = CeilAlign(tensorSlice[k].dataCount, perBlockCount);
      op(wDataLocal[offset], rDataLocal[0][offset], rDataLocal[1][offset], scalarDataType(scalar.GetValue(0)),
         dataCount);
      offset += dataCount;
    }
    dataOutQueue.EnQue(wDataLocal);
    for (int i = 0; i < rdDepth; i++) {
      dataInQueue[i].FreeTensor(rDataLocal[i]);
    }
  }
};

// Ternary Functors
template <typename T, int depth, int rdDepth>
struct TernaryOpListFunctor {
  template <typename Op>
  __aicore__ __inline__ void operator()(TQue<QuePosition::VECIN, BUFFER_NUM> (&dataInQueue)[MAX_INPUT_TENSOR_NUM],
                                        TQue<QuePosition::VECOUT, BUFFER_NUM>& dataOutQueue, uint32_t ubFactorElement,
                                        Op op) {
    LocalTensor<T> rDataLocal[rdDepth];
    LocalTensor<T> wDataLocal;
    for (int i = 0; i < rdDepth; i++) {
      rDataLocal[i] = dataInQueue[i].DeQue<T>();
    }
    wDataLocal = dataOutQueue.AllocTensor<T>();
    op(wDataLocal, rDataLocal[0], rDataLocal[1], rDataLocal[2], ubFactorElement);
    dataOutQueue.EnQue(wDataLocal);
    for (int i = 0; i < rdDepth; i++) {
      dataInQueue[i].FreeTensor(rDataLocal[i]);
    }
  }
};

template <typename T, typename scalarDataType, int depth, int rdDepth>
struct TernaryOpScalarFunctor {
  template <typename Op>
  __aicore__ __inline__ void operator()(TQue<QuePosition::VECIN, BUFFER_NUM> (&dataInQueue)[MAX_INPUT_TENSOR_NUM],
                                        TQue<QuePosition::VECOUT, BUFFER_NUM>& dataOutQueue, uint32_t dataCount, Op op,
                                        GM_ADDR scalar) {
    LocalTensor<T> rDataLocal[rdDepth];
    LocalTensor<T> wDataLocal;
    for (int i = 0; i < rdDepth; i++) {
      rDataLocal[i] = dataInQueue[i].DeQue<T>();
    }
    wDataLocal = dataOutQueue.AllocTensor<T>();
    GlobalTensor<scalarDataType> scalarLocal;
    scalarLocal.SetGlobalBuffer((__gm__ scalarDataType*)(scalar), 1);
    op(wDataLocal, rDataLocal[0], rDataLocal[1], rDataLocal[1], scalarDataType(scalarLocal.GetValue(0)), dataCount);
    dataOutQueue.EnQue(wDataLocal);
    for (int i = 0; i < rdDepth; i++) {
      dataInQueue[i].FreeTensor(rDataLocal[i]);
    }
  }
};

template <typename T, typename scalarDataType, const int depth, const int rdDepth>
struct TernaryOpScalarListFunctor {
  template <typename Op>
  __aicore__ __inline__ void operator()(TQue<QuePosition::VECIN, BUFFER_NUM> (&dataInQueue)[MAX_INPUT_TENSOR_NUM],
                                        TQue<QuePosition::VECOUT, BUFFER_NUM>& dataOutQueue, uint32_t sliceCount,
                                        TensorSlice_t (&tensorSlice)[FULL_TENSOR_NUM], Op op, GM_ADDR scalarList) {
    LocalTensor<T> rDataLocal[rdDepth];
    LocalTensor<T> wDataLocal;
    for (int i = 0; i < rdDepth; i++) {
      rDataLocal[i] = dataInQueue[i].DeQue<T>();
    }
    wDataLocal = dataOutQueue.AllocTensor<T>();
    uint32_t offset = 0;
    int32_t perBlockCount = BLOCK_LENGTH_BYTE / sizeof(T);
    for (uint32_t k = 0; k < sliceCount; k++) {
      GlobalTensor<scalarDataType> scalar;
      scalar.SetGlobalBuffer(((__gm__ scalarDataType*)(scalarList) + tensorSlice[k].tensorIdx), 1);
      uint32_t dataCount = CeilAlign(tensorSlice[k].dataCount, perBlockCount);
      op(wDataLocal[offset], rDataLocal[0][offset], rDataLocal[1][offset], rDataLocal[2][offset],
         scalarDataType(scalar.GetValue(0)), dataCount);
      offset += dataCount;
    }
    dataOutQueue.EnQue(wDataLocal);
    for (int i = 0; i < rdDepth; i++) {
      dataInQueue[i].FreeTensor(rDataLocal[i]);
    }
  }
};

}  // namespace foreachFunctors

#endif  // __FOR_EACH_FUNCTORS_H__