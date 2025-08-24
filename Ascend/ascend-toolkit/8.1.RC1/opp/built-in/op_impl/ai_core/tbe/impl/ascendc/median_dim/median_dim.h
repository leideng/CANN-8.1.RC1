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
 * \file median_dim.h
 * \brief
 */

#ifndef MEDIAN_DIM_H
#define MEDIAN_DIM_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

constexpr static int32_t TENSOR_COUNT = 3;
constexpr static int32_t BUFFER_NUM = 1;
constexpr static float FLOATZERO = 0.0f;
constexpr static float FLOATONE = 1.0f;
constexpr static int32_t INTZERO = 0;
constexpr static int32_t INTONE = 1;
constexpr static uint32_t UINTZERO = 0;
constexpr static int32_t BLOCKSIZE = 32;

namespace AscendC {
template<typename T, typename DTYPE>
class MedianDimKernel {
public:
__aicore__ inline MedianDimKernel() = delete;
__aicore__ inline MedianDimKernel(GM_ADDR inputTensors[TENSOR_COUNT], const MedianDimTilingData &tiling, TPipe* pipe)
{
    InitParams(inputTensors, tiling, pipe);
}

__aicore__ inline void Process()
{   
    for (int64_t idx = 0; idx < loops_; idx++) {
        CopyIn(idx, computeDimsInUB_, computeDimsInUB_, inputOffset_);
        Compute(idx, computeDimsInUB_);
        CopyOut(idx, computeDimsInUB_, inputOffset_); 
    }
    if (lastLoopDims_ != 0) {
        CopyIn(loops_, lastLoopDims_, alignA_, inputOffset_);
        Compute(loops_, alignA_);
        CopyOut(loops_, alignA_, inputOffset_); 
    }
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
};

private:
__aicore__ inline void InitParams(GM_ADDR inputTensors[TENSOR_COUNT], const MedianDimTilingData &tiling, TPipe* pipe)
{
    parallelDimSize_ = tiling.parallelDimSize;
    rankDimSize_ = tiling.rankDimSize;
    bigCore_ = tiling.bigCore;
    smallCore_ = tiling.smallCore;
    alignR_ = tiling.alignR;
    computeDimsInUB_ = tiling.computeDimsInUB;
    halfR_ = tiling.halfR;
    uintrankDimSize_ = static_cast<uint32_t>(rankDimSize_);
    intSizeofT = static_cast<uint32_t>(sizeof(T));

    auto blockIdx_ = GetBlockIdx();
    if (blockIdx_ < bigCore_) {
        coreRepTime_ = tiling.coreRepTime + INTONE;
        inputOffset_ = coreRepTime_ * blockIdx_;
    } else {
        coreRepTime_ = tiling.coreRepTime;
        inputOffset_ = (coreRepTime_ + INTONE) * bigCore_ + coreRepTime_ * (blockIdx_ - bigCore_);
    }

    lastLoopDims_ = coreRepTime_ % computeDimsInUB_;
    loops_ = coreRepTime_ / computeDimsInUB_;
    alignA_ = CeilAlign(lastLoopDims_, BLOCKSIZE / sizeof(T));
    
    selfGm_.SetGlobalBuffer((__gm__ T*)inputTensors[0]); 
    valuesGm_.SetGlobalBuffer((__gm__ T*)inputTensors[1]);
    indicesGm_.SetGlobalBuffer((__gm__ DTYPE*)inputTensors[2]);

    pipe->InitBuffer(selfQueue, BUFFER_NUM, computeDimsInUB_ * rankDimSize_ * sizeof(T));
    pipe->InitBuffer(valuesQueue, BUFFER_NUM, computeDimsInUB_ * sizeof(T));
    pipe->InitBuffer(indicesQueue, BUFFER_NUM, computeDimsInUB_ * sizeof(DTYPE)); 
    pipe->InitBuffer(tmpART1Queue, computeDimsInUB_ * rankDimSize_ * sizeof(T));
    pipe->InitBuffer(tmpART2Queue, computeDimsInUB_ * rankDimSize_ * sizeof(T));
    pipe->InitBuffer(nanMaskQueue, computeDimsInUB_ * alignR_ * sizeof(uint8_t));
    pipe->InitBuffer(tmpA1Queue, computeDimsInUB_ * sizeof(DTYPE));
}

__aicore__ inline void CopyIn(int32_t progress, int32_t currentDimsInUB, int32_t alignA, int64_t inputOffset)
{
    selfLocal = selfQueue.AllocTensor<T>();  
    for (int32_t i = 0; i < rankDimSize_; i++) {
        DataCopy(selfLocal[i * alignA], selfGm_[inputOffset + progress * computeDimsInUB_ + i * parallelDimSize_], alignA);
        PipeBarrier<PIPE_ALL>();
    }
    selfQueue.EnQue<T>(selfLocal);
}

__aicore__ inline void CopyOut(int32_t progress, int32_t currentDimsInUB, int64_t outputOffset)
{
    valuesLocal = valuesQueue.DeQue<T>();  
    indicesLocal = indicesQueue.DeQue<DTYPE>();
    DataCopy(valuesGm_[outputOffset + progress * computeDimsInUB_], valuesLocal, currentDimsInUB);
    DataCopy(indicesGm_[outputOffset + progress * computeDimsInUB_], indicesLocal, currentDimsInUB);
    valuesQueue.FreeTensor(valuesLocal);
    indicesQueue.FreeTensor(indicesLocal);
}

__aicore__ inline void SortAndGather(int32_t currentDimsInUB, uint32_t uintCurrentDimsInUB)
{   
    for (int i = 0; i < rankDimSize_ - 1; ++i) {
        for (int j = 0; j < rankDimSize_ - i - 1; ++j) {
            Compare(nanMask, selfLocal[(j + 1) * currentDimsInUB], selfLocal[(j + 1) * currentDimsInUB], CMPMODE::EQ, uintCurrentDimsInUB * alignR_);
            PipeBarrier<PIPE_V>();
            Select(tmpART1, nanMask, selfLocal[j * currentDimsInUB], selfLocal[(j + 1) * currentDimsInUB], SELMODE::VSEL_TENSOR_TENSOR_MODE, uintCurrentDimsInUB);
            PipeBarrier<PIPE_V>();
            Select(selfLocal[(j + 1) * currentDimsInUB], nanMask, selfLocal[(j + 1) * currentDimsInUB], selfLocal[j * currentDimsInUB], SELMODE::VSEL_TENSOR_TENSOR_MODE, uintCurrentDimsInUB);
            PipeBarrier<PIPE_V>();
            Select(selfLocal[j * currentDimsInUB], nanMask, selfLocal[j * currentDimsInUB], tmpART1, SELMODE::VSEL_TENSOR_TENSOR_MODE, uintCurrentDimsInUB);
            PipeBarrier<PIPE_V>();
            Select(tmpART1, nanMask, tmpART2[j * currentDimsInUB], tmpART2[(j + 1) * currentDimsInUB], SELMODE::VSEL_TENSOR_TENSOR_MODE, uintCurrentDimsInUB);
            PipeBarrier<PIPE_V>();
            Select(tmpART2[(j + 1) * currentDimsInUB], nanMask, tmpART2[(j + 1) * currentDimsInUB], tmpART2[j * currentDimsInUB], SELMODE::VSEL_TENSOR_TENSOR_MODE, uintCurrentDimsInUB);
            PipeBarrier<PIPE_V>();
            Select(tmpART2[j * currentDimsInUB], nanMask, tmpART2[j * currentDimsInUB], tmpART1, SELMODE::VSEL_TENSOR_TENSOR_MODE, uintCurrentDimsInUB);
            PipeBarrier<PIPE_V>();

            Compare(nanMask, selfLocal[j * currentDimsInUB], selfLocal[(j + 1) * currentDimsInUB], CMPMODE::LE, uintCurrentDimsInUB * alignR_);
            PipeBarrier<PIPE_V>();
            Select(tmpART1, nanMask, selfLocal[j * currentDimsInUB], selfLocal[(j + 1) * currentDimsInUB], SELMODE::VSEL_TENSOR_TENSOR_MODE, uintCurrentDimsInUB);
            PipeBarrier<PIPE_V>();
            Select(selfLocal[(j + 1) * currentDimsInUB], nanMask, selfLocal[(j + 1) * currentDimsInUB], selfLocal[j * currentDimsInUB], SELMODE::VSEL_TENSOR_TENSOR_MODE, uintCurrentDimsInUB);
            PipeBarrier<PIPE_V>();
            Select(selfLocal[j * currentDimsInUB], nanMask, selfLocal[j * currentDimsInUB], tmpART1, SELMODE::VSEL_TENSOR_TENSOR_MODE, uintCurrentDimsInUB);
            PipeBarrier<PIPE_V>();
            Select(tmpART1, nanMask, tmpART2[j * currentDimsInUB], tmpART2[(j + 1) * currentDimsInUB], SELMODE::VSEL_TENSOR_TENSOR_MODE, uintCurrentDimsInUB);
            PipeBarrier<PIPE_V>();
            Select(tmpART2[(j + 1) * currentDimsInUB], nanMask, tmpART2[(j + 1) * currentDimsInUB], tmpART2[j * currentDimsInUB], SELMODE::VSEL_TENSOR_TENSOR_MODE, uintCurrentDimsInUB);
            PipeBarrier<PIPE_V>();
            Select(tmpART2[j * currentDimsInUB], nanMask, tmpART2[j * currentDimsInUB], tmpART1, SELMODE::VSEL_TENSOR_TENSOR_MODE, uintCurrentDimsInUB);
            PipeBarrier<PIPE_V>();
        }
    }
    LocalTensor<uint32_t> addrUint = tmpA1.template ReinterpretCast<uint32_t>();
    PipeBarrier<PIPE_V>();
    Gather(valuesLocal, selfLocal, addrUint, UINTZERO, uintCurrentDimsInUB);
    PipeBarrier<PIPE_V>();
    Gather(tmpART1, tmpART2, addrUint, UINTZERO, uintCurrentDimsInUB);
    PipeBarrier<PIPE_V>();
    Cast(indicesLocal, tmpART1, RoundMode::CAST_TRUNC, uintCurrentDimsInUB);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void Compute(int32_t progress, int32_t currentDimsInUB) 
{   
    uint32_t uintCurrentDimsInUB = static_cast<uint32_t>(currentDimsInUB);
    selfLocal = selfQueue.DeQue<T>();                                               // (r,a) float
    valuesLocal = valuesQueue.AllocTensor<T>();                                     // (a) float
    indicesLocal = indicesQueue.AllocTensor<DTYPE>();                               // (a) int32
    // calculate num mask
    nanMask = nanMaskQueue.Get<uint8_t>();                                          // (r,a) uint8_t
    Compare(nanMask, selfLocal, selfLocal, CMPMODE::EQ, uintCurrentDimsInUB * alignR_); // (r,a) uint8_t cmp_result
    PipeBarrier<PIPE_V>();
    tmpART1 = tmpART1Queue.Get<T>();
    Duplicate<T>(tmpART1, FLOATZERO, rankDimSize_ * currentDimsInUB);
    PipeBarrier<PIPE_V>();
    Select(tmpART1, nanMask, tmpART1, FLOATONE, SELMODE::VSEL_TENSOR_SCALAR_MODE, uintrankDimSize_ * uintCurrentDimsInUB); // (a,r) fp32 cmp_result
    PipeBarrier<PIPE_V>();
    for (int32_t i = 0; i < rankDimSize_ - 1; i++) {
        Add(tmpART1, tmpART1, tmpART1[(i + 1) * currentDimsInUB], currentDimsInUB); // (a) float num_nan
        PipeBarrier<PIPE_V>();
    }
    CompareScalar(nanMask, tmpART1, FLOATZERO, CMPMODE::NE, uintCurrentDimsInUB * alignR_);   // (a)
    PipeBarrier<PIPE_V>();
    // calculate median index
    tmpART2 = tmpART2Queue.Get<T>();
    Duplicate<T>(tmpART2, rankDimSize_, currentDimsInUB);                              // (a)
    PipeBarrier<PIPE_V>();
    Sub(tmpART1, tmpART2, tmpART1, currentDimsInUB);
    PipeBarrier<PIPE_V>();
    Select(tmpART1, nanMask, tmpART1, halfR_, SELMODE::VSEL_TENSOR_SCALAR_MODE, uintCurrentDimsInUB);  // (a) fp32 cmp_result
    PipeBarrier<PIPE_V>();
    Cast(indicesLocal, tmpART1, RoundMode::CAST_TRUNC, uintCurrentDimsInUB);            // (a) int32 median_index
    PipeBarrier<PIPE_V>();
    Muls(indicesLocal, indicesLocal, currentDimsInUB, currentDimsInUB);                         // (a) int32 median_index * r
    PipeBarrier<PIPE_V>();
    tmpA1 = tmpA1Queue.Get<DTYPE>();
    CreateVecIndex(tmpA1, INTZERO, uintCurrentDimsInUB);                                // (a) int32 median_index * r + arrange_a
    PipeBarrier<PIPE_V>();
    Add(indicesLocal, indicesLocal, tmpA1, currentDimsInUB);                            // (a) int32 addr
    PipeBarrier<PIPE_V>();
    Muls(tmpA1, indicesLocal, intSizeofT, currentDimsInUB);                             // (a) int32 addr * bytes
    PipeBarrier<PIPE_V>();
    for (int i = 0; i < rankDimSize_; i++) {
        Duplicate<T>(tmpART2[i * currentDimsInUB], static_cast<float>(i), currentDimsInUB);
        PipeBarrier<PIPE_V>();
    }
    SortAndGather(currentDimsInUB, uintCurrentDimsInUB);
    selfQueue.FreeTensor(selfLocal);
    valuesQueue.EnQue<T>(valuesLocal);
    indicesQueue.EnQue<DTYPE>(indicesLocal);
}

private:
  TPipe* pipe;

  GlobalTensor<T> selfGm_;
  GlobalTensor<T> valuesGm_;
  GlobalTensor<DTYPE> indicesGm_;

  TQue<QuePosition::VECIN, BUFFER_NUM> selfQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> valuesQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> indicesQueue;

  TBuf<TPosition::VECCALC> nanMaskQueue;
  TBuf<TPosition::VECCALC> tmpART1Queue;
  TBuf<TPosition::VECCALC> tmpART2Queue;
  TBuf<TPosition::VECCALC> tmpA1Queue;

  LocalTensor<T> selfLocal;
  LocalTensor<T> valuesLocal;
  LocalTensor<DTYPE> indicesLocal;

  LocalTensor<T> tmpART1;
  LocalTensor<T> tmpART2;
  LocalTensor<DTYPE> tmpA1;
  LocalTensor<uint8_t> nanMask;

  int64_t parallelDimSize_ = 0;
  int32_t rankDimSize_ = 0;
  int32_t bigCore_ = 0;
  int32_t smallCore_ = 0;
  int32_t alignR_ = 0;
  int32_t alignA_ = 0;
  uint32_t uintrankDimSize_ = 0;
  float halfR_ = 0.0f;
  int32_t intSizeofT = 0;
  int64_t coreRepTime_ = 0;
  int64_t inputOffset_ = 0;
  int32_t computeDimsInUB_ = 0;
  int32_t lastLoopDims_ = 0;
  int64_t loops_ = 0;
};
}

#endif // MEDIAN_DIM_H