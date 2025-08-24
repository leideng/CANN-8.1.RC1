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
 * \file tome_merge.h
 * \brief
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;
namespace {
#if defined(__DAV_C220_VEC__)
// 12288 * 12 B available memory in 910B
constexpr uint64_t MAX_UB_MEMORY = 12288 * 12;
#else
// 12288 * 16 B available memory in 310P
constexpr uint64_t MAX_UB_MEMORY = 12288 * 16;
#endif
// 16 half nums equal to 32B
constexpr uint32_t BLOCK_SIZE = 16;
// Repeat size is 128
constexpr uint32_t REPEAT_SIZE = 128;
// Default repeat stride is 8
constexpr uint32_t DEFAULT_REPEAT_STRIDE = 8;
// Vector Mask of fp16 is 128
constexpr uint32_t FP16_MASK = 128;
// Vector Mask of fp32 is 64
constexpr uint32_t FP32_MASK = 64;
// Reduce Num is 8
constexpr uint64_t HEADS = 8;
// Repeat time is 255 at most
constexpr uint32_t DUP_REPEAT_MAX = 255;
constexpr int64_t MAX_BATCH = 64;
constexpr int64_t MAX_SEQLENA = 4096;
constexpr int64_t MAX_SEQLENB = 4096;

constexpr uint64_t DUPLICATE_MASK = 128;
constexpr int BLOCK_BYTES = 32;
constexpr uint64_t INT64_NUM_ONE_BLOCK = 4;
constexpr uint64_t FLOAT_NUM_ONE_BLOCK = 8;
}  // namespace

namespace TomeMergeND {
class TomeMerge {
 public:
  __aicore__ inline TomeMerge(uint64_t batch, uint64_t hiddenSize, uint64_t topR, uint64_t seqlenA, uint64_t seqlenB) {
    Check(batch, hiddenSize, topR, seqlenA, seqlenB);
    if (!valid) {
      return;
    }

    this->batch = batch;
    this->hiddenSize = hiddenSize;
    this->topR = topR;
    this->seqlenA = seqlenA;
    this->seqlenB = seqlenB;
    this->afterMergedLenA = seqlenA - topR;
    this->seqlenAD128 = (seqlenA + REPEAT_SIZE - 1) / REPEAT_SIZE * REPEAT_SIZE;
    this->seqlenBD128 = (seqlenB + REPEAT_SIZE - 1) / REPEAT_SIZE * REPEAT_SIZE;
#if defined(__DAV_M200__)
    // 2 tensors of length seqlenAD128, 1 tensor of length seqlenBD128
    this->usedMemory = seqlenBD128 * sizeof(float) + seqlenAD128 * sizeof(int64_t) * 2;
#else
    this->usedMemory = seqlenBD128 * sizeof(float);
#endif
    if (usedMemory > MAX_UB_MEMORY) {
      valid = false;
      return;
    }
    this->maxCacheMemory = MAX_UB_MEMORY - usedMemory;
    this->colsBase = (maxCacheMemory / sizeof(half) / hiddenSize / BLOCK_SIZE) * BLOCK_SIZE;
    if (colsBase == 0) {
      valid = false;
      return;
    }
    // 2 for half of colsBase
    this->colsBaseH = colsBase / 2;
    this->blockX = block_idx / HEADS;
    this->blockY = block_idx % HEADS;
  }

  __aicore__ inline void Init(GM_ADDR tokenA, GM_ADDR tokenB, GM_ADDR topkIndice, GM_ADDR argMax, GM_ADDR mergedToken,
                              GM_ADDR unreducedToken, GM_ADDR unreducedCount) {
    if (tokenA == nullptr || tokenB == nullptr || topkIndice == nullptr || argMax == nullptr ||
        mergedToken == nullptr || unreducedToken == nullptr || unreducedCount == nullptr) {
      valid = false;
    }
    if (!valid) {
      return;
    }
    tokenAGm.SetGlobalBuffer((__gm__ half*)tokenA + blockX * seqlenA * hiddenSize);
    tokenBGm.SetGlobalBuffer((__gm__ half*)tokenB + blockX * seqlenB * hiddenSize);
    argMaxGm.SetGlobalBuffer((__gm__ int64_t*)argMax + blockX * seqlenA);
    topkIndiceGm.SetGlobalBuffer((__gm__ int64_t*)topkIndice + blockX * seqlenA);

    mergedTokenGm.SetGlobalBuffer((__gm__ half*)mergedToken + blockX * afterMergedLenA * hiddenSize);
    unreducedTokenGm.SetGlobalBuffer((__gm__ half*)unreducedToken + blockX * HEADS * seqlenB * hiddenSize);
    unreducedCountGm.SetGlobalBuffer((__gm__ float*)unreducedCount + blockX * HEADS * seqlenB);

    pipe.InitBuffer(cacheInQueue, 1, colsBaseH * hiddenSize * sizeof(half));
    pipe.InitBuffer(cacheOutQueue, 1, colsBaseH * hiddenSize * sizeof(half));

    pipe.InitBuffer(countQueue, 1, (seqlenBD128 * sizeof(float)));
#if defined(__DAV_M200__)
    pipe.InitBuffer(indiceQueue, 1, (seqlenAD128 * sizeof(int64_t)));
    pipe.InitBuffer(argmaxQueue, 1, (seqlenAD128 * sizeof(int64_t)));
#endif
  }

  __aicore__ inline void Process() {
    if (!valid) {
      return;
    }
    /* stage 0: init unreducedToken and unreducedCount with zero */
    InitGm();
    LocalTensor<int64_t> indiceMemory;

#if defined(__DAV_M200__)
    indiceMemory = indiceQueue.DeQue<int64_t>();
#endif

    /* stage 1: copy unmerge part of tokenA and the whole tokenB to dst */
    CopyUnmerged(indiceMemory);
    /* stage 2: move and add data from tokenAGm to unreducedToken and update count */
    CopyReduced(indiceMemory);

#if defined(__DAV_M200__)
    indiceQueue.FreeTensor(indiceMemory);
#endif
  }

 private:
  __aicore__ inline void Check(uint64_t batch, uint64_t hiddenSize, uint64_t topR, uint64_t seqlenA, uint64_t seqlenB) {
    if (batch < 1 || batch > MAX_BATCH || seqlenA < 1 || seqlenA > MAX_SEQLENA || seqlenB < 1 ||
        seqlenB > MAX_SEQLENB || topR >= seqlenA) {
      valid = false;
    }
  }

  __aicore__ inline void InitGm() {
    if (!valid) {
      return;
    }

    if (blockY == 0) {
      uint64_t taskRepeat = seqlenB;
      uint64_t taskCurNum = taskRepeat;
      uint64_t colsRepeat = (taskCurNum + colsBaseH - 1) / colsBaseH;
      uint64_t colsRemain = taskCurNum % colsBaseH;

      for (uint64_t i = 0; i < colsRepeat; ++i) {
        uint64_t curCols = ((colsRemain != 0) && (i == colsRepeat - 1)) ? colsRemain : colsBaseH;
        LocalTensor<half> cacheMemoryIn = cacheInQueue.AllocTensor<half>();
        DataCopy(cacheMemoryIn, tokenBGm[i * colsBaseH * hiddenSize + blockY * taskRepeat * hiddenSize],
                 curCols * hiddenSize);
        cacheInQueue.EnQue(cacheMemoryIn);
        cacheMemoryIn = cacheInQueue.DeQue<half>();
        LocalTensor<half> cacheMemoryOut = cacheOutQueue.AllocTensor<half>();
        DataCopy(cacheMemoryOut, cacheMemoryIn, curCols * hiddenSize);
        cacheOutQueue.EnQue(cacheMemoryOut);
        cacheInQueue.FreeTensor(cacheMemoryIn);

        cacheMemoryOut = cacheOutQueue.DeQue<half>();
        DataCopy(unreducedTokenGm[i * colsBaseH * hiddenSize + blockY * taskRepeat * hiddenSize], cacheMemoryOut,
                 curCols * hiddenSize);
        cacheOutQueue.FreeTensor(cacheMemoryOut);
      }
      pipe_barrier(PIPE_ALL);
    } else {
      LocalTensor<half> cacheMemory = cacheOutQueue.AllocTensor<half>();
      Duplicate(cacheMemory, (half)0.0, (int32_t)(colsBaseH * hiddenSize));
      cacheOutQueue.EnQue(cacheMemory);
      cacheMemory = cacheOutQueue.DeQue<half>();
      uint64_t fillRepeat = (seqlenB + colsBaseH - 1) / colsBaseH;
      uint64_t fillRemain = seqlenB % colsBaseH;
      for (uint64_t i = 0; i < fillRepeat; ++i) {
        uint64_t curCol = ((fillRemain != 0) && (i == fillRepeat - 1)) ? fillRemain : colsBaseH;
        DataCopy(unreducedTokenGm[blockY * seqlenB * hiddenSize + i * colsBaseH * hiddenSize], cacheMemory,
                 curCol * hiddenSize);
      }

      cacheOutQueue.FreeTensor(cacheMemory);
    }

    float dupNum = 0.0;
    if (blockY == 0) {
      dupNum = 1.0;
    }
    LocalTensor<float> countMemory = countQueue.AllocTensor<float>();
    Duplicate(countMemory, dupNum, (int32_t)(seqlenBD128));
    countQueue.EnQue(countMemory);

#if defined(__DAV_M200__)
    uint64_t seqlenAAlign = (seqlenA + INT64_NUM_ONE_BLOCK - 1) / INT64_NUM_ONE_BLOCK * INT64_NUM_ONE_BLOCK;
    LocalTensor<int64_t> indiceMemory = indiceQueue.AllocTensor<int64_t>();
    DataCopy(indiceMemory, topkIndiceGm, seqlenAAlign);
    indiceQueue.EnQue(indiceMemory);
    pipe_barrier(PIPE_ALL);

    LocalTensor<int64_t> argmaxMemory = argmaxQueue.AllocTensor<int64_t>();
    DataCopy(argmaxMemory, argMaxGm, seqlenAAlign);
    argmaxQueue.EnQue(argmaxMemory);
    pipe_barrier(PIPE_ALL);
#endif
  }

  __aicore__ inline void CopyUnmerged(LocalTensor<int64_t> indiceMemory) {
    if (!valid) {
      return;
    }
    uint64_t taskRepeat = afterMergedLenA / HEADS;
    uint64_t taskCurNum = (blockY == (HEADS - 1)) ? (afterMergedLenA - (HEADS - 1) * taskRepeat) : taskRepeat;
    uint64_t colsRepeat = (taskCurNum + colsBaseH - 1) / colsBaseH;
    uint64_t colsRemain = taskCurNum % colsBaseH;

    for (uint64_t i = 0; i < colsRepeat; ++i) {
      uint64_t curCols = ((colsRemain != 0) && (i == colsRepeat - 1)) ? colsRemain : colsBaseH;
      LocalTensor<half> cacheMemoryIn = cacheInQueue.AllocTensor<half>();
      for (uint64_t j = 0; j < curCols; ++j) {
#if defined(__DAV_M200__)
        int64_t idxA = indiceMemory.GetValue(topR + blockY * taskRepeat + i * colsBaseH + j);
#else
        int64_t idxA = topkIndiceGm.GetValue(topR + blockY * taskRepeat + i * colsBaseH + j);
#endif
        DataCopy(cacheMemoryIn[j * hiddenSize], tokenAGm[idxA * hiddenSize], hiddenSize);
      }
      cacheInQueue.EnQue(cacheMemoryIn);
      cacheMemoryIn = cacheInQueue.DeQue<half>();
      LocalTensor<half> cacheMemoryOut = cacheOutQueue.AllocTensor<half>();
      DataCopy(cacheMemoryOut, cacheMemoryIn, curCols * hiddenSize);
      cacheOutQueue.EnQue(cacheMemoryOut);
      cacheInQueue.FreeTensor(cacheMemoryIn);

      cacheMemoryOut = cacheOutQueue.DeQue<half>();
      DataCopy(mergedTokenGm[i * colsBaseH * hiddenSize + blockY * taskRepeat * hiddenSize], cacheMemoryOut,
               curCols * hiddenSize);
      cacheOutQueue.FreeTensor(cacheMemoryOut);
    }
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void CopyReduced(LocalTensor<int64_t> indiceMemory) {
    if (!valid) {
      return;
    }
    uint64_t repeat = topR / HEADS;
    uint64_t curNum = (blockY == (HEADS - 1)) ? (topR - (HEADS - 1) * repeat) : repeat;
    uint64_t start = blockY * repeat;
    uint64_t end = start + curNum;

#if defined(__DAV_M200__)
    LocalTensor<int64_t> argmaxMemory = argmaxQueue.DeQue<int64_t>();
#endif

    LocalTensor<float> countMemory = countQueue.DeQue<float>();
    countUbuf = (__ubuf__ float*)countMemory.GetPhyAddr();
    for (uint64_t i = start; i < end; ++i) {
#if defined(__DAV_M200__)
      uint64_t idxA = indiceMemory.GetValue(i);
      uint64_t idxB = argmaxMemory.GetValue(idxA);
#else
      uint64_t idxA = topkIndiceGm.GetValue(i);
      uint64_t idxB = argMaxGm.GetValue(idxA);
#endif
      event_t eventIdMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
      SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
      WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
      LocalTensor<half> cacheMemoryIn = cacheInQueue.AllocTensor<half>();
      DataCopy(cacheMemoryIn, tokenAGm[idxA * hiddenSize], hiddenSize);
      DataCopy(cacheMemoryIn[hiddenSize], unreducedTokenGm[blockY * seqlenB * hiddenSize + idxB * hiddenSize],
               hiddenSize);
      cacheInQueue.EnQue(cacheMemoryIn);
      cacheMemoryIn = cacheInQueue.DeQue<half>();
      LocalTensor<half> cacheMemoryOut = cacheOutQueue.AllocTensor<half>();
      Add(cacheMemoryOut, cacheMemoryIn, cacheMemoryIn[hiddenSize], hiddenSize);
      cacheOutQueue.EnQue(cacheMemoryOut);
      cacheInQueue.FreeTensor(cacheMemoryIn);

      *(countUbuf + idxB) = *(countUbuf + idxB) + 1;
      pipe_barrier(PIPE_ALL);

      cacheMemoryOut = cacheOutQueue.DeQue<half>();
      DataCopy(unreducedTokenGm[blockY * seqlenB * hiddenSize + idxB * hiddenSize], cacheMemoryOut, hiddenSize);
      cacheOutQueue.FreeTensor(cacheMemoryOut);
    }

#if defined(__DAV_M200__)
    argmaxQueue.FreeTensor(argmaxMemory);
#endif

    pipe_barrier(PIPE_ALL);
    uint64_t seqlenBDownAlign = seqlenB / FLOAT_NUM_ONE_BLOCK * FLOAT_NUM_ONE_BLOCK;
    DataCopy(unreducedCountGm[blockY * seqlenB], countMemory, seqlenBDownAlign);
    if (seqlenBDownAlign != seqlenB) {
      countUbuf = (__ubuf__ float*)countMemory.GetPhyAddr();
      event_t eventIdMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
      SetFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
      WaitFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
      for (uint64_t i = 0; i < FLOAT_NUM_ONE_BLOCK; i++) {
        *(countUbuf + i) = *(countUbuf + i + seqlenB - FLOAT_NUM_ONE_BLOCK);
      }
      event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
      SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
      WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
      DataCopy(unreducedCountGm[blockY * seqlenB + seqlenB - FLOAT_NUM_ONE_BLOCK], countMemory, FLOAT_NUM_ONE_BLOCK);
    }
    countQueue.FreeTensor(countMemory);
    pipe_barrier(PIPE_ALL);
  }

 private:
  /* global memory address */
  GlobalTensor<half> tokenAGm;
  GlobalTensor<half> tokenBGm;
  GlobalTensor<half> mergedTokenGm;
  GlobalTensor<half> unreducedTokenGm;

  GlobalTensor<int64_t> topkIndiceGm;
  GlobalTensor<int64_t> argMaxGm;

  GlobalTensor<float> unreducedCountGm;

  /* variable */
  uint64_t batch = 0;
  uint64_t hiddenSize = 0;
  uint64_t seqlenA = 0;
  uint64_t seqlenB = 0;
  uint64_t topR = 0;
  uint64_t afterMergedLenA = 0;
  uint64_t usedMemory = 0;
  uint64_t maxCacheMemory = 0;

  uint64_t colsBase = 0;
  uint64_t colsBaseH = 0;
  uint64_t seqlenAD128 = 0;
  uint64_t seqlenBD128 = 0;

  uint32_t blockX = 0;
  uint32_t blockY = 0;

  /* ascendc variable */
  TPipe pipe;
  TQue<QuePosition::VECIN, 1> cacheInQueue;
  TQue<QuePosition::VECOUT, 1> cacheOutQueue;
  TQue<QuePosition::VECIN, 1> cacheQueue;
  TQue<QuePosition::VECIN, 1> countQueue;
  TQue<QuePosition::VECIN, 1> indiceQueue;
  TQue<QuePosition::VECIN, 1> argmaxQueue;
  __ubuf__ float* countUbuf;
#if defined(__DAV_M200__)
  __ubuf__ int64_t* indiceUbuf;
  __ubuf__ int64_t* argmaxUbuf;
#endif
  bool valid = true;
};
}  // namespace TomeMergeND
