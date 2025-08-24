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
 * \file tome_unmerge.hpp
 * \brief
 */
#ifndef __TOME_UNMERGE_H__
#define __TOME_UNMERGE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace {
constexpr uint64_t MAX_UB_MEMORY = 12288 * 16;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t REPEAT_SIZE = 128;
constexpr uint32_t DEFAULT_REPEAT_STRIDE = 8;
constexpr uint64_t kHeads = 4;
constexpr int64_t MAX_BATCH = 64;
constexpr int64_t MAX_SEQLENA = 4096;
constexpr int64_t MAX_SEQLENB = 4096;
constexpr uint64_t INT64_NUM_ONE_BLOCK = 4;
}  // namespace

using namespace AscendC;
class TomeUnmerge {
 public:
  __aicore__ inline TomeUnmerge(uint64_t batch, uint64_t hiddenSize, uint64_t topR, uint64_t seqLenA,
                                uint64_t seqLenB) {
    Check(batch, hiddenSize, topR, seqLenA, seqLenB);
    if (!valid) {
      return;
    }

    this->batch = batch;
    this->hiddenSize = hiddenSize;
    this->topR = topR;
    this->seqLenA = seqLenA;
    this->seqLenB = seqLenB;
    this->seqLenAD128 = (seqLenA + REPEAT_SIZE - 1) / REPEAT_SIZE * REPEAT_SIZE;
    this->seqLenBD128 = (seqLenB + REPEAT_SIZE - 1) / REPEAT_SIZE * REPEAT_SIZE;
#if defined(__DAV_M200__)
    this->usedMemory = (seqLenAD128 * 3 + seqLenBD128) * sizeof(int64_t);
#else
    this->usedMemory = 0;
#endif
    if (usedMemory > MAX_UB_MEMORY) {
      valid = false;
      return;
    }
    this->maxCacheMemory = MAX_UB_MEMORY - usedMemory;
    this->afterMergedLenA = seqLenA - topR;
    this->afterMergedLen = this->afterMergedLenA + this->seqLenB;
    this->colsBase = (maxCacheMemory / sizeof(half) / hiddenSize / BLOCK_SIZE) * BLOCK_SIZE;
    if (colsBase == 0) {
      valid = false;
      return;
    }
    srcBatchOffset = this->afterMergedLen * hiddenSize;
    curBurstLen = hiddenSize * sizeof(half) / 32; //32 block size
  }

  __aicore__ inline void Init(__gm__ uint8_t* attention, __gm__ uint8_t* oriIndiceA, __gm__ uint8_t* oriIndiceB,
                              __gm__ uint8_t* topkIndice, __gm__ uint8_t* argMax, __gm__ uint8_t* unZipToken) {
    if (attention == nullptr || oriIndiceA == nullptr || oriIndiceB == nullptr || topkIndice == nullptr ||
        argMax == nullptr || unZipToken == nullptr) {
      valid = false;
    }
    if (!valid) {
      return;
    }

    curBatch = block_idx / (taskPerBatch * kHeads);
    curTask = (block_idx % (taskPerBatch * kHeads)) / kHeads;

    attenOutGm.SetGlobalBuffer((__gm__ half *)attention + curBatch * srcBatchOffset);
    oriIndiceAGm.SetGlobalBuffer((__gm__ int64_t *)oriIndiceA + curBatch * seqLenA);
    oriIndiceBGm.SetGlobalBuffer((__gm__ int64_t *)oriIndiceB + curBatch * seqLenB);
    topkIndiceGm.SetGlobalBuffer((__gm__ int64_t *)topkIndice + curBatch * seqLenA);
    argmaxGm.SetGlobalBuffer((__gm__ int64_t *)argMax + curBatch * seqLenA);
    unZipTokenGm.SetGlobalBuffer((__gm__ half *)unZipToken + curBatch * (seqLenA + seqLenB) * hiddenSize);

    pipe.InitBuffer(cacheInQueue, 1, colsBase * hiddenSize * sizeof(half) / 2);
    pipe.InitBuffer(cacheOutQueue, 1, colsBase * hiddenSize * sizeof(half) / 2);
#if defined(__DAV_M200__)
    pipe.InitBuffer(oriIndiceAQueue, 1, (seqLenAD128 * sizeof(int64_t)));
    pipe.InitBuffer(oriIndiceBQueue, 1, (seqLenBD128 * sizeof(int64_t)));
    pipe.InitBuffer(topkIndiceQueue, 1, (seqLenAD128 * sizeof(int64_t)));
    pipe.InitBuffer(argmaxQueue, 1, (seqLenAD128 * sizeof(int64_t)));
#endif
  }

  __aicore__ inline void Process() {
    if (!valid) {
      return;
    }
#if defined(__DAV_M200__)
    uint64_t seqlenAAlign = (seqLenA + INT64_NUM_ONE_BLOCK - 1) / INT64_NUM_ONE_BLOCK * INT64_NUM_ONE_BLOCK;
    uint64_t seqlenBAlign = (seqLenB + INT64_NUM_ONE_BLOCK - 1) / INT64_NUM_ONE_BLOCK * INT64_NUM_ONE_BLOCK;
    LocalTensor<int64_t> oriIndiceAMemory = oriIndiceAQueue.AllocTensor<int64_t>();
    DataCopy(oriIndiceAMemory, oriIndiceAGm, seqlenAAlign);
    oriIndiceAUbuf = (__ubuf__ int64_t*)oriIndiceAMemory.GetPhyAddr();
    LocalTensor<int64_t> oriIndiceBMemory = oriIndiceBQueue.AllocTensor<int64_t>();
    DataCopy(oriIndiceBMemory, oriIndiceBGm, seqlenBAlign);
    oriIndiceBUbuf = (__ubuf__ int64_t*)oriIndiceBMemory.GetPhyAddr();
    LocalTensor<int64_t> topkIndiceMemory = topkIndiceQueue.AllocTensor<int64_t>();
    DataCopy(topkIndiceMemory, topkIndiceGm, seqlenAAlign);
    topkIndiceUbuf = (__ubuf__ int64_t*)topkIndiceMemory.GetPhyAddr();
    LocalTensor<int64_t> argmaxMemory = argmaxQueue.AllocTensor<int64_t>();
    DataCopy(argmaxMemory, argmaxGm, seqlenAAlign);
    argmaxUbuf = (__ubuf__ int64_t*)argmaxMemory.GetPhyAddr();
    pipe_barrier(PIPE_ALL);
#endif

    if (curTask == 0) {
      ProcessUnmergeA();
    } else if (curTask == 1) {
      ProcessMergedB();
    } else if (curTask == 2) { //2 means second task
      ProcessMergedA();
    } else {
      ProcessMergedHalfA();
    }
  }

 private:
  __aicore__ inline void Check(uint64_t batch, uint64_t hiddenSize, uint64_t topR, uint64_t seqLenA, uint64_t seqLenB) {
    if (batch < 1 || batch > MAX_BATCH ||
        seqLenA < 1 || seqLenA > MAX_SEQLENA ||
        seqLenB < 1 || seqLenB > MAX_SEQLENB ||
        topR >= seqLenA || hiddenSize <= 0) {
      valid = false;
    }
  }

  __aicore__ inline void ProcessUnmergeA() {
    if (!valid) {
      return;
    }

    uint64_t colsBaseH = colsBase / 2;
    uint64_t colsRepeat = (afterMergedLenA + colsBaseH - 1) / colsBaseH;
    uint64_t colsRemain = afterMergedLenA % colsBaseH == 0 ? colsBaseH : afterMergedLenA % colsBaseH;

    uint64_t curHead = (block_idx % (taskPerBatch * kHeads)) % kHeads;
    uint64_t repeatPerHead = (colsRepeat + kHeads - 1) / kHeads;
    uint64_t src = curHead * repeatPerHead;
    uint64_t dst = ((src + repeatPerHead) > colsRepeat) ? colsRepeat : src + repeatPerHead;
    for (uint64_t i = src; i < dst; ++i) {
      uint64_t curCols = (i == colsRepeat - 1) ? colsRemain : colsBaseH;
      uint64_t idx = i % 2;
      LocalTensor<half> cacheIn = cacheInQueue.AllocTensor<half>();
      DataCopy(cacheIn, attenOutGm[i * colsBaseH * hiddenSize],
               curCols * hiddenSize);
      cacheInQueue.EnQue(cacheIn);
      cacheIn = cacheInQueue.DeQue<half>();
      LocalTensor<half> cacheOut = cacheOutQueue.AllocTensor<half>();
      DataCopy(cacheOut, cacheIn, curCols * hiddenSize);
      cacheInQueue.FreeTensor(cacheIn);
      cacheOutQueue.EnQue(cacheOut);
      cacheOut = cacheOutQueue.DeQue<half>();
      for (uint64_t j = 0; j < curCols; ++j) {
#if defined(__DAV_M200__)
        beforeMergedIndicesA = *(topkIndiceUbuf + topR + i * colsBaseH + j);
        oriIndicesA = *(oriIndiceAUbuf + beforeMergedIndicesA);
#else
        beforeMergedIndicesA = *(topkIndiceGm.GetPhyAddr() + topR + i * colsBaseH + j);
        oriIndicesA = *(oriIndiceAGm.GetPhyAddr() + beforeMergedIndicesA);
#endif
        DataCopy(unZipTokenGm[oriIndicesA * hiddenSize], cacheOut[j * hiddenSize],
                 hiddenSize);
      }
      cacheOutQueue.FreeTensor(cacheOut);
    }
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void ProcessMergedB() {
    if (!valid) {
      return;
    }

    uint64_t colsBaseH = colsBase / 2;
    uint64_t colsRepeat = (seqLenB + colsBaseH - 1) / colsBaseH;
    uint64_t colsRemain = seqLenB % colsBaseH == 0 ? colsBaseH : seqLenB % colsBaseH;

    uint64_t curHead = (block_idx % (taskPerBatch * kHeads)) % kHeads;
    uint64_t repeatPerHead = (colsRepeat + kHeads - 1) / kHeads;
    uint64_t src = curHead * repeatPerHead;
    uint64_t dst = ((src + repeatPerHead) > colsRepeat) ? colsRepeat : src + repeatPerHead;
    for (uint64_t i = src; i < dst; ++i) {
      uint64_t curCols = (i == colsRepeat - 1) ? colsRemain : colsBaseH;
      uint64_t idx = i % 2;
      LocalTensor<half> cacheIn = cacheInQueue.AllocTensor<half>();
      DataCopy(cacheIn,
               attenOutGm[afterMergedLenA * hiddenSize + i * colsBaseH * hiddenSize],
               curCols * hiddenSize);
      cacheInQueue.EnQue(cacheIn);
      cacheIn = cacheInQueue.DeQue<half>();
      LocalTensor<half> cacheOut = cacheOutQueue.AllocTensor<half>();
      DataCopy(cacheOut, cacheIn, curCols * hiddenSize);
      cacheInQueue.FreeTensor(cacheIn);
      cacheOutQueue.EnQue(cacheOut);
      cacheOut = cacheOutQueue.DeQue<half>();
      for (uint64_t j = 0; j < curCols; ++j) {
        beforeMergedIndicesB = i * colsBaseH + j;
#if defined(__DAV_M200__)
        oriIndicesB = *(oriIndiceBUbuf + beforeMergedIndicesB);
#else
        oriIndicesB = *(oriIndiceBGm.GetPhyAddr() + beforeMergedIndicesB);
#endif
        DataCopy(unZipTokenGm[oriIndicesB * hiddenSize],
                 cacheOut[j * hiddenSize],
                 hiddenSize);
      }
      cacheOutQueue.FreeTensor(cacheOut);
    }
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void ProcessMergedA() {
    if (!valid) {
      return;
    }

    uint64_t colsBaseH = colsBase / 2;
    uint64_t colsRepeat = (topR / 2 + colsBaseH - 1) / colsBaseH;
    uint64_t colsRemain = (topR / 2) % colsBaseH == 0 ? colsBaseH : (topR / 2) % colsBaseH;

    uint64_t curHead = (block_idx % (taskPerBatch * kHeads)) % kHeads;
    uint64_t repeatPerHead = (colsRepeat + kHeads - 1) / kHeads;
    uint64_t src = curHead * repeatPerHead;
    uint64_t dst = ((src + repeatPerHead) > colsRepeat) ? colsRepeat : src + repeatPerHead;
    for (uint64_t i = src; i < dst; ++i) {
      uint64_t curCols = (i == colsRepeat - 1) ? colsRemain : colsBaseH;
      uint64_t idx = i % 2;

      for (uint64_t j = 0; j < curCols; ++j) {
#if defined(__DAV_M200__)
        beforeMergedIndicesA = *(topkIndiceUbuf + i * colsBaseH + j);
        afterMergedIndicesA = *(argmaxUbuf + beforeMergedIndicesA);
        oriIndicesA = *(oriIndiceAUbuf + beforeMergedIndicesA);
#else
        beforeMergedIndicesA = *(topkIndiceGm.GetPhyAddr() + i * colsBaseH + j);
        afterMergedIndicesA = *(argmaxGm.GetPhyAddr() + beforeMergedIndicesA);
        oriIndicesA = *(oriIndiceAGm.GetPhyAddr() + beforeMergedIndicesA);
#endif
        LocalTensor<half> cacheIn = cacheInQueue.AllocTensor<half>();
        DataCopy(cacheIn,
                 attenOutGm[(afterMergedLenA + afterMergedIndicesA) * hiddenSize],
                 hiddenSize);
        cacheInQueue.EnQue(cacheIn);
        cacheIn = cacheInQueue.DeQue<half>();
        LocalTensor<half> cacheOut = cacheOutQueue.AllocTensor<half>();
        DataCopy(cacheOut, cacheIn, hiddenSize);
        cacheInQueue.FreeTensor(cacheIn);
        cacheOutQueue.EnQue(cacheOut);
        pipe_barrier(PIPE_ALL);
        cacheOut = cacheOutQueue.DeQue<half>();
        DataCopy(unZipTokenGm[oriIndicesA * hiddenSize],
                 cacheOut,
                 hiddenSize);
        cacheOutQueue.FreeTensor(cacheOut);
      }
      pipe_barrier(PIPE_ALL);
    }
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void ProcessMergedHalfA() {
    if (!valid) {
      return;
    }
    uint64_t halfLen = (topR % 2 == 0) ? (topR / 2) : (topR / 2 + 1);
    uint64_t colsBaseH = colsBase / 2;
    uint64_t colsRepeat = (halfLen + colsBaseH - 1) / colsBaseH;
    uint64_t colsRemain = halfLen % colsBaseH == 0 ? colsBaseH : halfLen % colsBaseH;

    uint64_t curHead = (block_idx % (taskPerBatch * kHeads)) % kHeads;
    uint64_t repeatPerHead = (colsRepeat + kHeads - 1) / kHeads;
    uint64_t src = curHead * repeatPerHead;
    uint64_t dst = ((src + repeatPerHead) > colsRepeat) ? colsRepeat : src + repeatPerHead;
    for (uint64_t i = src; i < dst; ++i) {
      uint64_t curCols = (i == colsRepeat - 1) ? colsRemain : colsBaseH;
      uint64_t idx = i % 2;

      for (uint64_t j = 0; j < curCols; ++j) {
#if defined(__DAV_M200__)
        beforeMergedIndicesA = *(topkIndiceUbuf + i * colsBaseH + j + (topR / 2));
        afterMergedIndicesA = *(argmaxUbuf + beforeMergedIndicesA);
        oriIndicesA = *(oriIndiceAUbuf + beforeMergedIndicesA);
#else
        beforeMergedIndicesA = *(topkIndiceGm.GetPhyAddr() + i * colsBaseH + j + (topR / 2));
        afterMergedIndicesA = *(argmaxGm.GetPhyAddr() + beforeMergedIndicesA);
        oriIndicesA = *(oriIndiceAGm.GetPhyAddr() + beforeMergedIndicesA);
#endif
        LocalTensor<half> cacheIn = cacheInQueue.AllocTensor<half>();
        DataCopy(cacheIn,
                 attenOutGm[(afterMergedLenA + afterMergedIndicesA) * hiddenSize],
                 hiddenSize);
        cacheInQueue.EnQue(cacheIn);
        cacheIn = cacheInQueue.DeQue<half>();
        LocalTensor<half> cacheOut = cacheOutQueue.AllocTensor<half>();
        DataCopy(cacheOut, cacheIn, hiddenSize);
        cacheInQueue.FreeTensor(cacheIn);
        cacheOutQueue.EnQue(cacheOut);
        pipe_barrier(PIPE_ALL);
        cacheOut = cacheOutQueue.DeQue<half>();
        DataCopy(unZipTokenGm[oriIndicesA * hiddenSize],
                 cacheOut,
                 hiddenSize);
        cacheOutQueue.FreeTensor(cacheOut);
      }
      pipe_barrier(PIPE_ALL);
    }
    pipe_barrier(PIPE_ALL);
  }

 private:
  GlobalTensor<half> attenOutGm;
  GlobalTensor<int64_t> oriIndiceAGm;
  GlobalTensor<int64_t> oriIndiceBGm;
  GlobalTensor<int64_t> topkIndiceGm;
  GlobalTensor<int64_t> argmaxGm;

  GlobalTensor<half> unZipTokenGm;

  uint64_t batch = 1;
  uint64_t hiddenSize = 1;
  uint64_t seqLenA = 0;
  uint64_t seqLenB = 0;
  uint64_t seqLenAD128 = 0;
  uint64_t seqLenBD128 = 0;
  uint64_t usedMemory = 0;
  uint64_t maxCacheMemory = 0;
  uint64_t topR = 0;

  uint64_t afterMergedLenA = 0;
  uint64_t afterMergedLen = 0;
  uint64_t batchOffsetCurrent = 0;
  uint64_t colsBase = 0;
  uint64_t taskPerBatch = 4;
  uint64_t curBatch = 0;
  uint64_t srcBatchOffset = 0;
  uint64_t curTask = 0;
  uint64_t curBurstLen = 0;

  uint64_t beforeMergedIndicesA = 0;
  uint64_t oriIndicesA = 0;
  uint64_t oriIndicesB = 0;
  uint64_t beforeMergedIndicesB = 0;
  uint64_t afterMergedIndicesA = 0;

  TPipe pipe;
  TQue<QuePosition::VECIN, 1> cacheInQueue;
  TQue<QuePosition::VECOUT, 1> cacheOutQueue;
  TQue<QuePosition::VECIN, 1> oriIndiceAQueue;
  TQue<QuePosition::VECIN, 1> oriIndiceBQueue;
  TQue<QuePosition::VECIN, 1> topkIndiceQueue;
  TQue<QuePosition::VECIN, 1> argmaxQueue;
#if defined(__DAV_M200__)
  __ubuf__ int64_t* oriIndiceAUbuf;
  __ubuf__ int64_t* oriIndiceBUbuf;
  __ubuf__ int64_t* topkIndiceUbuf;
  __ubuf__ int64_t* argmaxUbuf;
#endif
  bool valid = true;
};

#endif //TOME_UNMERGE_H