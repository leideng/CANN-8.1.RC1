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
 * \file multi_tensor_apply.h
 * \brief
 */
#ifndef __MULTI_TENSOR_APPLY_H__
#define __MULTI_TENSOR_APPLY_H__

#include "foreach_functors.h"

using namespace AscendC;
using namespace foreachFunctors;
namespace multiTensorApply {

constexpr int32_t BUFFER_NUM = 2;

#define KERNEL_LOG1(format, ...)

template <const int32_t depth, const int32_t rdDepth, int32_t updateArgIndex, typename TilingData,
          typename TensorDataType, typename scalarDataType, typename U, typename... ArgTypes>
class multiTensorApply {
 public:
  __aicore__ inline multiTensorApply(){};
  __aicore__ inline void Init(GM_ADDR tensorDataList[depth], GM_ADDR workspace, TilingData* tilingData);
  __aicore__ inline void Process(U callable, ArgTypes... args);

 private:
  __aicore__ inline void CopyIn();
  __aicore__ inline void CopyOut();
  __aicore__ inline __gm__ TensorDataType* GetTensorAddr(int32_t iterDepth, uint16_t index);
  template <typename T1, typename T2>
  __aicore__ inline T1 GetMinValue(T1 a, T2 b) {
    return a <= b ? a : b;
  };

 private:
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> dataInQueue[MAX_INPUT_TENSOR_NUM];
  TQue<QuePosition::VECOUT, BUFFER_NUM> dataOutQueue;
  GlobalTensor<TensorDataType> tensorListInGM[depth];  // input:0~depth-2; output:depth
  GlobalTensor<scalarDataType> scalarListInGM;
  GM_ADDR tensorDataList[depth];
  uint32_t blockIdx = 0;
  uint32_t perBlockCount = 0;
  TilingData* tilingData;
  uint32_t curSiceCnt = 0;
  TensorSlice_t curTensorSlice[FULL_TENSOR_NUM];
};

template <int32_t depth, int32_t rdDepth, int32_t updateArgIndex, typename TilingData, typename TensorDataType,
          typename scalarDataType, typename U, typename... ArgTypes>
__aicore__ inline void multiTensorApply<depth, rdDepth, updateArgIndex, TilingData, TensorDataType, scalarDataType, U,
                                        ArgTypes...>::Init(GM_ADDR tensorDataList[depth], GM_ADDR workspace,
                                                           TilingData* tilingData) {
  this->blockIdx = GetBlockIdx();
  perBlockCount = BLOCK_LENGTH_BYTE / sizeof(TensorDataType);
  for (uint32_t i = 0; i < depth; i++) {
    this->tensorDataList[i] = tensorDataList[i];
  }
  this->tilingData = tilingData;
  for (int i = 0; i < rdDepth; i++) {
    this->pipe.InitBuffer(dataInQueue[i], BUFFER_NUM, tilingData->ubFactorElement * sizeof(TensorDataType));
  }
  this->pipe.InitBuffer(dataOutQueue, BUFFER_NUM, tilingData->ubFactorElement * sizeof(TensorDataType));
  this->curSiceCnt = 0;
}

template <int32_t depth, int32_t rdDepth, int32_t updateArgIndex, typename TilingData, typename TensorDataType,
          typename scalarDataType, typename U, typename... ArgTypes>
__aicore__ inline void multiTensorApply<depth, rdDepth, updateArgIndex, TilingData, TensorDataType, scalarDataType, U,
                                        ArgTypes...>::Process(U callable, ArgTypes... args) {
  uint16_t tensorStart = tilingData->listStartIdx[blockIdx];
  uint16_t tensorEnd = tilingData->listEndIdx[blockIdx];
  int64_t curCount = 0;
  for (uint16_t i = tensorStart; i <= tensorEnd; i++) {
    int64_t curStartOffset = 0;
    int64_t curEndOffset = tilingData->tensorDataCountList[i] - 1;
    int64_t dataCount = 0;
    if (i == tensorStart) {
      curStartOffset = tilingData->tensorStartOffset[blockIdx];
    }
    if (i == tensorEnd) {
      curEndOffset = tilingData->tensorEndOffset[blockIdx];
    }

    dataCount = curEndOffset - curStartOffset + 1;
    uint32_t copyTimes = CeilDiv(dataCount + curCount, tilingData->ubFactorElement);
    KERNEL_LOG1("ubFactorElement:%d, block:%d start:%d,startoffset:%d", tilingData->ubFactorElement, blockIdx,
                tensorStart, curStartOffset);
    KERNEL_LOG1(",end:%d,endoffset:%d,dataCount:%ld,copyTimes%d\n", tensorEnd, curEndOffset, dataCount, copyTimes);
    for (uint32_t j = 0; j < copyTimes; j++) {
      KERNEL_LOG1("First dataCount:%d,%d,%d,%d\n", tilingData->ubFactorElement, curCount, dataCount, blockIdx);
      int64_t tempCount = GetMinValue(tilingData->ubFactorElement - curCount, curEndOffset - curStartOffset + 1);
      KERNEL_LOG1("dataCount:%d,%d,%d\n", tempCount, curEndOffset - curStartOffset + 1, blockIdx);
      curTensorSlice[curSiceCnt].tensorIdx = i;
      curTensorSlice[curSiceCnt].offset = curStartOffset;
      curTensorSlice[curSiceCnt].dataCount = tempCount;
      KERNEL_LOG1("keep dataCount:%d,%d,%d,%d,%d\n", i, j, curTensorSlice[curSiceCnt].offset,
                  curTensorSlice[curSiceCnt].dataCount, blockIdx);
      curSiceCnt++;
      curCount += CeilAlign(tempCount, perBlockCount);
      KERNEL_LOG1("curCount:%d,%d,%d\n", curCount, curSiceCnt, i);
      if (curCount >= tilingData->ubFactorElement || curSiceCnt == FULL_TENSOR_NUM || i == tensorEnd) {
        KERNEL_LOG1("kernel in:%d,%d,%d,%d,%d\n", curCount, curSiceCnt, i, tensorEnd, blockIdx);
        curStartOffset += tempCount;
        curCount = 0;
        CopyIn();
        if constexpr (IsSameType<U,
                                 BinaryOpScalarListFunctor<TensorDataType, scalarDataType, depth, rdDepth>>::value ||
                      IsSameType<U, UnaryOpScalarListFunctor<TensorDataType, scalarDataType, depth, rdDepth>>::value ||
                      IsSameType<U,
                                 TernaryOpScalarListFunctor<TensorDataType, scalarDataType, depth, rdDepth>>::value) {
          callable(dataInQueue, dataOutQueue, curSiceCnt, curTensorSlice, args...);
        } else {
          callable(dataInQueue, dataOutQueue, tilingData->ubFactorElement, args...);
        }
        CopyOut();
        curSiceCnt = 0;
      }
    }
  }
}

template <int32_t depth, int32_t rdDepth, int32_t updateArgIndex, typename TilingData, typename TensorDataType,
          typename scalarDataType, typename U, typename... ArgTypes>
__aicore__ inline void multiTensorApply<depth, rdDepth, updateArgIndex, TilingData, TensorDataType, scalarDataType, U,
                                        ArgTypes...>::CopyIn() {
  LocalTensor<TensorDataType> dataLocal[rdDepth];
  for (uint32_t i = 0; i < rdDepth; i++) {
    dataLocal[i] = dataInQueue[i].AllocTensor<TensorDataType>();
  }
  uint32_t offset = 0;
  DataCopyParams copyParams = {1, 0, 0, 0};
  DataCopyPadParams padParams = {true, 0, 0, 0};
  for (uint32_t i = 0; i < curSiceCnt; i++) {
    int64_t alignDataCount = CeilAlign(curTensorSlice[i].dataCount, perBlockCount);
    copyParams.blockLen = curTensorSlice[i].dataCount * sizeof(TensorDataType);
    padParams.rightPadding = alignDataCount - curTensorSlice[i].dataCount;
    padParams.paddingValue = GetScalarBitcodeValue((TensorDataType)0);
    for (int j = 0; j < rdDepth; j++) {
      KERNEL_LOG1("copy in:%d, --%d,%d,%d,%d\n", curTensorSlice[i].tensorIdx, curTensorSlice[i].offset,
                  curTensorSlice[i].dataCount, j, this->blockIdx);
      tensorListInGM[j].SetGlobalBuffer(GetTensorAddr(j, curTensorSlice[i].tensorIdx));
      DataCopyPad(dataLocal[j][offset], tensorListInGM[j][curTensorSlice[i].offset], copyParams, padParams);
    }
    offset += alignDataCount;
  }
  for (uint32_t j = 0; j < rdDepth; j++) {
    dataInQueue[j].EnQue(dataLocal[j]);
  }
}

template <int32_t depth, int32_t rdDepth, int32_t updateArgIndex, typename TilingData, typename TensorDataType,
          typename scalarDataType, typename U, typename... ArgTypes>
__aicore__ inline void multiTensorApply<depth, rdDepth, updateArgIndex, TilingData, TensorDataType, scalarDataType, U,
                                        ArgTypes...>::CopyOut() {
  LocalTensor<TensorDataType> dataLocal = dataOutQueue.DeQue<TensorDataType>();
  uint32_t offset = 0;
  DataCopyParams copyParams{1, 0, 0, 0};
  for (uint64_t i = 0; i < curSiceCnt; i++) {
    KERNEL_LOG1("copy out tensorIdx:%d, %d, %d, %d, %d\n", curTensorSlice[i].tensorIdx, curTensorSlice[i].offset,
                curTensorSlice[i].dataCount, updateArgIndex, this->blockIdx);
    tensorListInGM[updateArgIndex].SetGlobalBuffer(GetTensorAddr(updateArgIndex, curTensorSlice[i].tensorIdx));
    copyParams.blockLen = curTensorSlice[i].dataCount * sizeof(TensorDataType);
    DataCopyPad(tensorListInGM[updateArgIndex][curTensorSlice[i].offset], dataLocal[offset], copyParams);
    offset += CeilAlign(curTensorSlice[i].dataCount, perBlockCount);
  }
  dataOutQueue.FreeTensor(dataLocal);
}

template <int32_t depth, int32_t rdDepth, int32_t updateArgIndex, typename TilingData, typename TensorDataType,
          typename scalarDataType, typename U, typename... ArgTypes>
__aicore__ inline __gm__ TensorDataType*
multiTensorApply<depth, rdDepth, updateArgIndex, TilingData, TensorDataType, scalarDataType, U,
                 ArgTypes...>::GetTensorAddr(int32_t iterDepth, uint16_t index) {
  __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorDataList[iterDepth]);
  uint64_t tensorPtrOffset = *dataAddr;
  __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
  return reinterpret_cast<__gm__ TensorDataType*>(*(tensorPtr + index));
}

template <int32_t depth, int32_t rdDepth, int32_t updateArgIndex, typename TilingData, typename TensorDataType,
          typename scalarDataType, typename U, typename... ArgTypes>
__aicore__ inline void multiTensorApplyKernel(GM_ADDR tensorDataList[depth], GM_ADDR workspace, TilingData* tilingData,
                                              U callable, ArgTypes... args) {
  // Hand the chunk information to the user-supplied functor to process however it likes.
  class multiTensorApply<depth, rdDepth, updateArgIndex, TilingData, TensorDataType, scalarDataType, U, ArgTypes...>
      mta;
  mta.Init(tensorDataList, workspace, tilingData);
  mta.Process(callable, args...);
}

}  // namespace multiTensorApply

#endif  // __MULTI_TENSOR_APPLY_H__
