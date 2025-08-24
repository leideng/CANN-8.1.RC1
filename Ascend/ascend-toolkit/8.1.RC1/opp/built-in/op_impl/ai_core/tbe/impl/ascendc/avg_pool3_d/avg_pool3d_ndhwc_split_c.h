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
 * \file avg_pool3d_ndhwc_split_c.h
 * \brief
 */

#ifndef AVG_POOL3D_NDHWC_SPLIT_C_H_
#define AVG_POOL3D_NDHWC_SPLIT_C_H_

#include "kernel_operator.h"
#include "avg_pool3d_common.h"

namespace AvgPool3d {
template <typename T, int32_t QUEUE_DEPTH>
class KernelAvgPool3dSplitC {
public:
  __aicore__ inline KernelAvgPool3dSplitC() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* tiling, TPipe* pipe);
  __aicore__ inline void Process();

private:
  __aicore__ inline void InitTiling(const AvgPool3DTilingData* tiling);
  __aicore__ inline void CopyIn(int64_t offset, int64_t len);
  __aicore__ inline void CopyOut(int64_t offset, int64_t len);
  __aicore__ inline void ReduceMeanWindow(int64_t outputPointIdx);
  __aicore__ inline void ReduceSumWindow(
    const Index& index, LocalTensor<float>& sumBufLocal, int64_t nOffset, int64_t cOffset, int64_t len);
  
  TPipe* pipe;
  TQue<QuePosition::VECIN, QUEUE_DEPTH> inputQueue;
  TQue<QuePosition::VECOUT, QUEUE_DEPTH> outputQueue;

  TBuf<TPosition::VECCALC> sumBuf;
  LocalTensor<float> sumBufLocal;

  GlobalTensor<T> inputGlobal;
  GlobalTensor<T> outputGlobal;

  int64_t inC;
  int64_t tileC;
  int64_t outputPointNum;
  int64_t outputPointOffset;

  PoolShape inputShape;
  PoolShape outputShape;

  int64_t indexBufLen;
  IndexBuffer indexBuf;
  PoolParameter poolParam;

  uint32_t numPerBlock;
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::InitTiling(const AvgPool3DTilingData* tiling) {
  inputShape = PoolShape(tiling->inN, tiling->inC, tiling->inD, tiling->inH, tiling->inW);
  outputShape = PoolShape(tiling->inN, tiling->inC, tiling->outD, tiling->outH, tiling->outW);

  poolParam = PoolParameter(tiling->kD, tiling->kH, tiling->kW, tiling->dD, tiling->dH, tiling->dW,
                            tiling->pD, tiling->pH, tiling->pW, tiling->divisorOverride, tiling->countIncludePad);

  indexBuf.SetComputeParameter(outputShape, inputShape, poolParam);
  
  numPerBlock = GetDataBlockSizeInBytes() / sizeof(T);
  inC = tiling->inC;
  tileC = tiling->tileC;

  outputPointNum = GetBlockIdx() < tiling->formerNum ? tiling->formerLength : tiling->tailLength;
  outputPointOffset = GetBlockIdx() < tiling->formerNum
    ? tiling->formerLength * GetBlockIdx()
    : tiling->formerNum * tiling->formerLength + tiling->tailLength * (GetBlockIdx() - tiling->formerNum);

  indexBufLen = tiling->indexBufLen;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::CopyIn(int64_t offset, int64_t len) {
  LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
#if __CCE_AICORE__ < 220
  if constexpr (std::is_same_v<T, float>) {
    DataCopy(inputLocal, inputGlobal[offset], len);
  } else {
    DataCopy(inputLocal[tileC], inputGlobal[offset], len);
  }
#else
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
  if constexpr (std::is_same_v<T, float>) {
    DataCopyPad(inputLocal, inputGlobal[offset], copyParams, padParams);
  } else {
    DataCopyPad(inputLocal[tileC], inputGlobal[offset], copyParams, padParams);
  }
#endif
  inputQueue.EnQue(inputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::CopyOut(int64_t offset, int64_t len) {
  LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();
#if __CCE_AICORE__ < 220
  DataCopy(outputGlobal[offset], outputLocal, len);
#else
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
  DataCopyPad(outputGlobal[offset], outputLocal, copyParams);
#endif
  outputQueue.FreeTensor(outputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::ReduceSumWindow(
    const Index& index, LocalTensor<float>& sumBufLocal, int64_t nOffset, int64_t cOffset, int64_t len) {
  int64_t dstart = index.D.start;
  int64_t dend = index.D.end;
  int64_t hstart = index.H.start;
  int64_t hend = index.H.end;
  int64_t wstart = index.W.start;
  int64_t wend = index.W.end;

  int64_t startOffset = nOffset * inputShape.strideN + cOffset;
  for (int64_t id = dstart; id < dend; ++id) {
    int64_t dOffset = id * inputShape.strideD;
    for (int64_t ih = hstart; ih < hend; ++ih) {
      int64_t hOffset = ih * inputShape.strideH;
      for (int64_t iw = wstart; iw < wend; ++iw) {
        CopyIn(startOffset + (dOffset + hOffset + iw * inputShape.strideW) * inC, len);

        LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
        if constexpr (std::is_same_v<T, float>) {
          Add(sumBufLocal, sumBufLocal, inputLocal, len);
        } else {
          Cast(inputLocal.template ReinterpretCast<float>(), inputLocal[tileC], RoundMode::CAST_NONE, len);
          Add(sumBufLocal, sumBufLocal, inputLocal.template ReinterpretCast<float>(), len);
        }
        inputQueue.FreeTensor(inputLocal);
      }
    }
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::ReduceMeanWindow(int64_t outputPointIdx) {
  Index index;
  indexBuf.GetIndex(outputPointIdx, index);

  int64_t poolSize = poolParam.divisorOverride ?
                     poolParam.divisorOverride : index.D.poolSize * index.H.poolSize * index.W.poolSize;
  float factor = 1.0f / static_cast<float>(poolSize);

  SToVSync();

  int64_t cLoop = (inC + tileC - 1) / tileC;
  int64_t cOffset = 0;
  for (int64_t i = 0; i < cLoop; ++i) {
    int64_t count = i < cLoop - 1 ? tileC : inC - (cLoop - 1) * tileC;

    Duplicate(sumBufLocal, 0.0f, count);

    ReduceSumWindow(index, sumBufLocal, outputPointIdx / outputShape.strideC, cOffset, count);
    Muls(sumBufLocal, sumBufLocal, factor, count);

    LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
      DataCopy(outputLocal, sumBufLocal, AlignUp(count, numPerBlock));
    } else {
      Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, count);
    }
    outputQueue.EnQue(outputLocal);

    CopyOut(outputPointIdx * inC + cOffset, count);

    cOffset += count;
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* tiling, TPipe* pipe) {
  InitTiling(tiling);

  inputGlobal.SetGlobalBuffer((__gm__ T*)x);
  outputGlobal.SetGlobalBuffer((__gm__ T*)y);

  pipe->InitBuffer(inputQueue, QUEUE_DEPTH, tileC * sizeof(float));
  pipe->InitBuffer(outputQueue, QUEUE_DEPTH, tileC * sizeof(T));

  pipe->InitBuffer(sumBuf, tileC * sizeof(float));
  sumBufLocal = sumBuf.Get<float>();

  indexBuf.Init(pipe, indexBufLen);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::Process() {
  for (int64_t outputPointIdx = outputPointOffset;
       outputPointIdx < outputPointOffset + outputPointNum; ++outputPointIdx) {
    ReduceMeanWindow(outputPointIdx);
  }
}

} // namespace AvgPool3d

#endif // AVG_POOL3D_NDHWC_SPLIT_C_H_
