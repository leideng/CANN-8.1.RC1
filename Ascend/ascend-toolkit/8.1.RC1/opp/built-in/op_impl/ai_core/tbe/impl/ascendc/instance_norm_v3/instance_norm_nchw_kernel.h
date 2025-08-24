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
 * \file instance_norm_nchw_kernel.h
 * \brief
 */

#ifndef __INSTANCE_NORM_NCHW_KERNEL_H_
#define __INSTANCE_NORM_NCHW_KERNEL_H_

#include "instance_norm_nchw_base.h"

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelInstanceNormNCHW : public KernelInstanceNormNCHWBase<T, TILING_KEY, BUFFER_NUM> {
 public:
  __aicore__ inline KernelInstanceNormNCHW(TPipe* pipe) {
    Ppipe = pipe;
  }

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance,
                              GM_ADDR workspace, const InstanceNormV3TilingData* tiling) {
    this->InitBaseParams(tiling);
    this->InitInGlobalTensors(x, gamma, beta);
    this->InitOutGlobalTensors(y, mean, variance);

    Ppipe->InitBuffer(inDataQue, BUFFER_NUM, this->ubFactor * sizeof(T));   // D * 2
    Ppipe->InitBuffer(outDataQue, BUFFER_NUM, this->ubFactor * sizeof(T));  // D * 2

    Ppipe->InitBuffer(outReduceBuf, 2 * this->cAxisFactor * sizeof(float));  // D * 2
    Ppipe->InitBuffer(xBufFp32, this->ubFactor * sizeof(float));             // D * 4
    Ppipe->InitBuffer(yBufFp32, this->ubFactor * sizeof(float));             // D * 4
    Ppipe->InitBuffer(gammaBuf, this->cAxisFactor * sizeof(float));          // D * 4
    Ppipe->InitBuffer(betaBuf, this->cAxisFactor * sizeof(float));           // D * 4
  }

  __aicore__ inline void Process() {
    uint32_t cLoops = this->cAxis / this->cAxisFactor;
    uint32_t cTails = this->cAxis % this->cAxisFactor;
    uint32_t cGmOffset = 0;
    for (uint32_t cBlockIdx = 0; cBlockIdx < cLoops; ++cBlockIdx) {
      this->ProcessCBlock(cGmOffset, this->cAxisFactor);
      cGmOffset += this->cAxisFactor;
    }
    if (likely(cTails > 0)) {
      this->ProcessCBlock(cGmOffset, cTails);
    }
  }

 private:
  __aicore__ inline void ProcessCBlock(uint32_t cGmOffset, uint32_t cNums) {
    LocalTensor<float> gammaLocal = gammaBuf.Get<float>();
    LocalTensor<float> betaLocal = betaBuf.Get<float>();

    uint32_t gmOffset = 0;
    uint32_t reduceGmOffset = cGmOffset;

    float gammaScalar;
    float betaScalar;

    CopyInGammaBeta(gammaLocal, betaLocal, cGmOffset, cNums);

    for (uint32_t nIdx = 0; nIdx < this->nLoops; ++nIdx) {
      gmOffset = cGmOffset + nIdx * this->cAxis * this->reduceNums;
      for (uint32_t cIdx = 0; cIdx < cNums; ++cIdx) {
        gammaScalar = gammaLocal.GetValue(cIdx);
        betaScalar = betaLocal.GetValue(cIdx);
        CopyInHW(gmOffset, this->reduceNums);
        ComputeNormSingleHW(gammaScalar, betaScalar, cIdx);
        CopyOut(gmOffset, this->reduceNums);
        gmOffset += this->reduceNums;
      }
      CopyOutMeanVar(reduceGmOffset, cNums);
      reduceGmOffset += this->cAxis;
    }
  }

  __aicore__ inline void CopyOutMeanVar(uint32_t gmOffset, uint32_t size) {
    LocalTensor<float> meanVarTensor = outReduceBuf.Get<float>();
    LocalTensor<T> meanVarTensorOutType = meanVarTensor.ReinterpretCast<T>();
    LocalTensor<T> meanOut = meanVarTensorOutType[0];
    LocalTensor<T> varOut = meanVarTensorOutType[this->cAxisFactor];

    LocalTensor<T> fakeTensor = outDataQue.template AllocTensor<T>();

    if constexpr (!is_same<T, float>::value) {
      // total size of meanvar is 2 cfactor
      Cast(meanVarTensorOutType, meanVarTensor, RoundMode::CAST_NONE, this->cAxisFactor * 2);
      pipe_barrier(PIPE_V);
    } else {
    }
    outDataQue.EnQue(fakeTensor);
    LocalTensor<T> fakeOut = outDataQue.template DeQue<T>();

    DataCopyCustomUB2GM(this->meanGm[gmOffset], meanOut, size);
    DataCopyCustomUB2GM(this->varianceGm[gmOffset], varOut, size);

    outDataQue.FreeTensor(fakeOut);
  }

  __aicore__ inline void ComputeNormSingleHW(float gammaScalar, float betaScalar, int32_t cIdx) {
    LocalTensor<float> xFp32Tensor = xBufFp32.Get<float>();
    LocalTensor<float> yFp32Tensor = yBufFp32.Get<float>();

    LocalTensor<float> meanVarTensor = outReduceBuf.Get<float>();
    LocalTensor<float> meanTensor = meanVarTensor[0];
    LocalTensor<float> varTensor = meanVarTensor[this->cAxisFactor];

    event_t eventSV;
    event_t eventVS;

    Muls(yFp32Tensor, xFp32Tensor, this->avgFactor, this->reduceNums);  // yFp32Tensor <- x / N
    pipe_barrier(PIPE_V);
    ReduceSum(yFp32Tensor, yFp32Tensor, yFp32Tensor, this->reduceNums);

    eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, eventVS);
    wait_flag(PIPE_V, PIPE_S, eventVS);
    float aveLocalTemp = yFp32Tensor.GetValue(0);
    meanTensor.SetValue(cIdx, aveLocalTemp);

    eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, eventSV);
    wait_flag(PIPE_S, PIPE_V, eventSV);

    Adds(xFp32Tensor, xFp32Tensor, aveLocalTemp * -1, this->reduceNums);  // xFp32Tensor <- x - E(x)
    pipe_barrier(PIPE_V);
    Mul(yFp32Tensor, xFp32Tensor, xFp32Tensor, this->reduceNums);  // yFp32Tensor <- (x - E(x)) ** 2
    pipe_barrier(PIPE_V);
    Muls(yFp32Tensor, yFp32Tensor, this->avgFactor, this->reduceNums);  // yFp32Tensor <- (x - E(x)) ** 2 / N
    pipe_barrier(PIPE_V);

    ReduceSum(yFp32Tensor, yFp32Tensor, yFp32Tensor, this->reduceNums);

    eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, eventVS);
    wait_flag(PIPE_V, PIPE_S, eventVS);
    float varLocalTemp = yFp32Tensor.GetValue(0);
    varTensor.SetValue(cIdx, varLocalTemp);
    float rstdLocalTemp = 1 / sqrt(varLocalTemp + this->eps);
    eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, eventSV);
    wait_flag(PIPE_S, PIPE_V, eventSV);

    Muls(xFp32Tensor, xFp32Tensor, rstdLocalTemp, this->reduceNums);  // xFp32Tensor <- (x - E(x)) * rstd
    pipe_barrier(PIPE_V);

    Muls(yFp32Tensor, xFp32Tensor, gammaScalar, this->reduceNums);  // yFp32Tensor <- (x - E(x)) * rstd * gamma
    pipe_barrier(PIPE_V);

    LocalTensor<T> yLocal = outDataQue.template AllocTensor<T>();
    if constexpr (is_same<T, float>::value) {
      Adds(yLocal, yFp32Tensor, betaScalar, this->reduceNums);  // yLocal <- (x - E(x)) * rstd * gamma + beta
    } else {
      Adds(xFp32Tensor, yFp32Tensor, betaScalar, this->reduceNums);  // xFp32Tensor <- (x - E(x)) * rstd * gamma + beta
      pipe_barrier(PIPE_V);
      Cast(yLocal, xFp32Tensor, RoundMode::CAST_NONE, this->reduceNums);
    }
    pipe_barrier(PIPE_V);
    outDataQue.EnQue(yLocal);
  }

  __aicore__ inline void CopyOut(uint32_t gmOffset, uint32_t size) {
    LocalTensor<T> yOut = outDataQue.template DeQue<T>();
    DataCopyCustomUB2GM(this->yGm[gmOffset], yOut, size);
    outDataQue.FreeTensor(yOut);
  }

  __aicore__ inline void CopyInGammaBeta(LocalTensor<float>& gammaLocal, LocalTensor<float>& betaLocal,
                                         uint32_t cGmOffset, uint32_t size) {
    if constexpr (is_same<T, float>::value) {
      DataCopyCustomGM2UB(gammaLocal, this->gammaGm[cGmOffset], size);
      DataCopyCustomGM2UB(betaLocal, this->betaGm[cGmOffset], size);
      event_t eventMTE2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
      set_flag(PIPE_MTE2, PIPE_S, eventMTE2S);
      wait_flag(PIPE_MTE2, PIPE_S, eventMTE2S);
    } else {
      auto gammaLocalHalf = gammaLocal.ReinterpretCast<half>()[this->cAxisFactor];
      auto betaLocalHalf = betaLocal.ReinterpretCast<half>()[this->cAxisFactor];
      DataCopyCustomGM2UB(gammaLocalHalf, this->gammaGm[cGmOffset], size);
      DataCopyCustomGM2UB(betaLocalHalf, this->betaGm[cGmOffset], size);

      event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
      set_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
      wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V);

      Cast(gammaLocal, gammaLocalHalf, RoundMode::CAST_NONE, size);
      Cast(betaLocal, betaLocalHalf, RoundMode::CAST_NONE, size);
      event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
      set_flag(PIPE_V, PIPE_S, eventVS);
      wait_flag(PIPE_V, PIPE_S, eventVS);
      pipe_barrier(PIPE_V);
    }
  }

  __aicore__ inline void CopyInHW(uint32_t gmOffset, uint32_t size) {
    LocalTensor<T> xIn = inDataQue.template AllocTensor<T>();
    DataCopyCustomGM2UB(xIn, this->xGm[gmOffset], size);
    inDataQue.EnQue(xIn);
    LocalTensor<T> xLocal = inDataQue.template DeQue<T>();
    LocalTensor<float> xFp32Tensor = xBufFp32.Get<float>();
    if constexpr (is_same<T, float>::value) {
      Adds(xFp32Tensor, xLocal, (float)0.0, size);
    } else {
      Cast(xFp32Tensor, xLocal, RoundMode::CAST_NONE, size);
    }
    pipe_barrier(PIPE_V);
    inDataQue.FreeTensor(xLocal);
  }

 private:
  TPipe* Ppipe = nullptr;
  TQue<QuePosition::VECIN, BUFFER_NUM> inDataQue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outDataQue;

  TBuf<TPosition::VECCALC> xBufFp32;
  TBuf<TPosition::VECCALC> yBufFp32;

  TBuf<TPosition::VECCALC> gammaBuf;
  TBuf<TPosition::VECCALC> betaBuf;

  TBuf<TPosition::VECCALC> outReduceBuf;
};

#endif  // __INSTANCE_NORM_NCHW_KERNEL_H_
