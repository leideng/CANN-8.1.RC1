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
 * \file quantize_add_layer_norm_single_row_kernel.h
 * \brief
 */

#ifndef __QUANTIZE_ADD_LAYER_NORM_SINGLE_ROW_KERNEL_H_
#define __QUANTIZE_ADD_LAYER_NORM_SINGLE_ROW_KERNEL_H_

#include "quantize_add_layer_norm_base.h"

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelQuantizeAddLayerNormSingleRow {
#define IS_MUL_MODE ((TILING_KEY % 10) == 2)
 public:
  __aicore__ inline KernelQuantizeAddLayerNormSingleRow(TPipe* pipe) {
    Ppipe = pipe;
  }

  __aicore__ inline void Init(__gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* gamma, __gm__ uint8_t* beta,
                              __gm__ uint8_t* bias, __gm__ uint8_t* scales, __gm__ uint8_t* offsets, __gm__ uint8_t* y,
                              __gm__ uint8_t* x, __gm__ uint8_t* workspace, uint32_t numCore_, uint32_t numLastDim_,
                              uint32_t numFirstDim_, uint32_t firstDimPerCore_, uint32_t firstDimPerCoreTail_,
                              float eps_, float aveNum_) {
    numCore = numCore_;
    numLastDim = numLastDim_;
    numFirstDim = numFirstDim_;
    nlFirstDimPerCore = firstDimPerCore_;
    lFirstDimPerCore = firstDimPerCoreTail_;
    aveNum = aveNum_;
    eps = eps_;
    if (block_idx != numCore - 1) {
      rowWork = nlFirstDimPerCore;
    } else {
      rowWork = lFirstDimPerCore;
    }
    numLastDimAligned = numLastDim;
    if (ROUND_UP32(numLastDim * sizeof(T)) != numLastDim * sizeof(T)) {
      lastDimPad = true;
      numLastDimAligned = ROUND_UP32(numLastDim * sizeof(T)) / sizeof(T);
    }

    gmOffset_ = nlFirstDimPerCore * numLastDim;
    x1Gm.SetGlobalBuffer((__gm__ T*)(x1) + block_idx * gmOffset_);
    x2Gm.SetGlobalBuffer((__gm__ T*)(x2) + block_idx * gmOffset_);
    gammaGm.SetGlobalBuffer((__gm__ T*)gamma);
    betaGm.SetGlobalBuffer((__gm__ T*)beta);
    biasGm.SetGlobalBuffer((__gm__ T*)bias);
    scalesGm.SetGlobalBuffer((__gm__ T*)scales);
    if (offsets != nullptr) {
      offsetsGm.SetGlobalBuffer((__gm__ T*)offsets);
      hasOffset = true;
    }
    workspaceGm.SetGlobalBuffer((__gm__ float*)workspace + block_idx * 2 * numLastDim);

    yGm.SetGlobalBuffer((__gm__ int8_t*)(y) + block_idx * gmOffset_);
    xGm.SetGlobalBuffer((__gm__ T*)(x) + block_idx * gmOffset_);

    // single row
    Ppipe->InitBuffer(rowInQue, BUFFER_NUM, numLastDimAligned * sizeof(T));
    Ppipe->InitBuffer(rowOutQue, BUFFER_NUM, numLastDimAligned * sizeof(T));
    Ppipe->InitBuffer(quantizeOutQue, BUFFER_NUM, numLastDimAligned * sizeof(int8_t));
    Ppipe->InitBuffer(tmpBuf, 2 * numLastDimAligned * sizeof(float));
    Ppipe->InitBuffer(gammaBetaBuf, 2 * numLastDimAligned * sizeof(T));
  }

  __aicore__ inline void Process() {
    CopyInBetaGammaBias();
    for (int32_t row_idx = 0; row_idx < rowWork; ++row_idx) {
      CopyInAddSingleRow(row_idx);
      precision_compute_single_row(row_idx);
    }
  }

 private:
  __aicore__ inline void CopyInAddSingleRow(int32_t row_idx) {
    LocalTensor<float> tmpTensors = tmpBuf.Get<float>();

    LocalTensor<float> x1Fp32 = tmpTensors[0];
    LocalTensor<float> x2Fp32 = tmpTensors[numLastDimAligned];

    LocalTensor<T> x1In, x2In;

    if constexpr (is_same<float, T>::value) {
      x1In = x1Fp32;
      x2In = x2Fp32;
    } else {
      x1In = x2Fp32.ReinterpretCast<T>()[0];
      x2In = x2Fp32.ReinterpretCast<T>()[numLastDimAligned];
    }

    LocalTensor<T> biasIn = rowInQue.template AllocTensor<T>();
    uint32_t gm_offset = row_idx * numLastDim;
    DataCopyEx(x1In, x1Gm[gm_offset], numLastDim);
    DataCopyEx(x2In, x2Gm[gm_offset], numLastDim);

    rowInQue.EnQue(biasIn);
    auto biasTensor = rowInQue.template DeQue<T>();

    if constexpr (is_same<float, T>::value) {
      Add(x1In, x1In, x2In, numLastDim);
      pipe_barrier(PIPE_V);
      Add(x1In, x1In, biasTensor, numLastDim);
      auto xOutTensor = rowOutQue.template AllocTensor<T>();
      pipe_barrier(PIPE_V);
      Adds(xOutTensor, x1In, ZERO, numLastDim);
      rowOutQue.template EnQue<T>(xOutTensor);
      auto xOut = rowOutQue.template DeQue<T>();
      DataCopyEx(xGm[gm_offset], xOut, numLastDim);
      rowOutQue.FreeTensor(xOut);
    } else {
      Cast(x1Fp32, x1In, RoundMode::CAST_NONE, numLastDim);
      pipe_barrier(PIPE_V);
      Cast(x2Fp32, x2In, RoundMode::CAST_NONE, numLastDim);
      pipe_barrier(PIPE_V);
      Add(x1Fp32, x1Fp32, x2Fp32, numLastDim);
      pipe_barrier(PIPE_V);
      Cast(x2Fp32, biasTensor, RoundMode::CAST_NONE, numLastDim);
      pipe_barrier(PIPE_V);
      Add(x1Fp32, x1Fp32, x2Fp32, numLastDim);
      auto xOutTensor = rowOutQue.template AllocTensor<T>();
      pipe_barrier(PIPE_V);
      Cast(xOutTensor, x1Fp32, RoundMode::CAST_RINT, numLastDim);
      rowOutQue.template EnQue<T>(xOutTensor);
      auto xOut = rowOutQue.template DeQue<T>();
      DataCopyEx(xGm[gm_offset], xOut, numLastDim);
      rowOutQue.FreeTensor(xOut);
    }
    pipe_barrier(PIPE_V);
    rowInQue.FreeTensor(biasTensor);
  }

  __aicore__ inline void precision_compute_single_row(int32_t row_idx) {
    LocalTensor<float> tmpTensors = tmpBuf.Get<float>();
    LocalTensor<T> gammaBetaTensor = gammaBetaBuf.Get<T>();

    LocalTensor<float> xTensor = tmpTensors[0];
    LocalTensor<float> tmpTensor = tmpTensors[numLastDimAligned];
    LocalTensor<T> gammaTensor = gammaBetaTensor[0];
    LocalTensor<T> betaTensor = gammaBetaTensor[numLastDimAligned];

    Muls(tmpTensor, xTensor, aveNum, numLastDim);
    pipe_barrier(PIPE_V);
    float aveLocalTemp = ReduceSumFP32(tmpTensor, numLastDim);
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, eventSV);
    wait_flag(PIPE_S, PIPE_V, eventSV);
    Adds(xTensor, xTensor, -1 * aveLocalTemp, numLastDim);
    pipe_barrier(PIPE_V);
    Mul(tmpTensor, xTensor, xTensor, numLastDim);
    pipe_barrier(PIPE_V);
    Muls(tmpTensor, tmpTensor, aveNum, numLastDim);
    pipe_barrier(PIPE_V);
    float varLocalTemp = ReduceSumFP32(tmpTensor, numLastDim);
    float rstdLocalTemp = 1 / sqrt(varLocalTemp + eps);
    eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, eventSV);
    wait_flag(PIPE_S, PIPE_V, eventSV);

    LocalTensor<T> inTensor = rowOutQue.template AllocTensor<T>();
    event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    set_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
    DataCopyEx(inTensor, scalesGm, numLastDim);

    Cast(tmpTensor, gammaTensor, RoundMode::CAST_NONE, numLastDim);
    Muls(xTensor, xTensor, rstdLocalTemp, numLastDim);
    pipe_barrier(PIPE_V);

    Mul(xTensor, tmpTensor, xTensor, numLastDim);
    pipe_barrier(PIPE_V);
    Cast(tmpTensor, betaTensor, RoundMode::CAST_NONE, numLastDim);
    pipe_barrier(PIPE_V);
    Add(xTensor, tmpTensor, xTensor, numLastDim);

    // // small operator action
    if constexpr (!is_same<float, T>::value) {
      pipe_barrier(PIPE_V);
      Cast(tmpTensor.ReinterpretCast<T>(), xTensor, RoundMode::CAST_RINT, numLastDim);
      pipe_barrier(PIPE_V);
      Cast(xTensor, tmpTensor.ReinterpretCast<T>(), RoundMode::CAST_NONE, numLastDim);
    }
    pipe_barrier(PIPE_V);

    event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    set_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
    wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
    Cast(tmpTensor, inTensor, RoundMode::CAST_NONE, numLastDim);
    pipe_barrier(PIPE_V);

    event_t eventVMTE2;
    if (hasOffset) {
      event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
      set_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
      wait_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
      DataCopyEx(inTensor, offsetsGm, numLastDim);
    }

    if constexpr (IS_MUL_MODE) {
      Mul(xTensor, xTensor, tmpTensor, numLastDim);
    } else {
      Div(xTensor, xTensor, tmpTensor, numLastDim);
    }
    pipe_barrier(PIPE_V);

    if (hasOffset) {
      eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
      set_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
      wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
      Cast(tmpTensor, inTensor, RoundMode::CAST_NONE, numLastDim);
      pipe_barrier(PIPE_V);
      Add(xTensor, xTensor, tmpTensor, numLastDim);
      pipe_barrier(PIPE_V);
    }

    rowOutQue.FreeTensor(inTensor);
    eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    set_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
    wait_flag(PIPE_V, PIPE_MTE2, eventVMTE2);

    LocalTensor<int8_t> yLocal = quantizeOutQue.template AllocTensor<int8_t>();
    Cast(xTensor.ReinterpretCast<int32_t>(), xTensor, RoundMode::CAST_RINT, numLastDim);
    pipe_barrier(PIPE_V);
    SetDeqScale((half)1.000000e+00f);
    pipe_barrier(PIPE_V);
    Cast(xTensor.ReinterpretCast<half>(), xTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, numLastDim);
    pipe_barrier(PIPE_V);
    Cast(yLocal, xTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, numLastDim);
    quantizeOutQue.EnQue(yLocal);
    auto yOut = quantizeOutQue.template DeQue<int8_t>();
    DataCopyEx(yGm[row_idx * numLastDim], yOut, numLastDim);
    quantizeOutQue.FreeTensor(yOut);
  }

  __aicore__ inline void CopyInBetaGammaBias() {
    LocalTensor<T> gammaBetaTensor = gammaBetaBuf.Get<T>();

    LocalTensor<T> gammaIn = gammaBetaTensor[0];
    LocalTensor<T> betaIn = gammaBetaTensor[numLastDimAligned];
    LocalTensor<T> biasIn = rowInQue.template AllocTensor<T>();

    DataCopyEx(gammaIn, gammaGm, numLastDim);
    DataCopyEx(betaIn, betaGm, numLastDim);
    DataCopyEx(biasIn, biasGm, numLastDim);

    rowInQue.EnQue(biasIn);
    auto biasTensor = rowInQue.template DeQue<T>();

    rowInQue.FreeTensor(biasTensor);
  }

 private:
  TPipe* Ppipe = nullptr;
  TQue<QuePosition::VECIN, BUFFER_NUM> rowInQue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> rowOutQue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> quantizeOutQue;

  TBuf<TPosition::VECCALC> tmpBuf, gammaBetaBuf;

  GlobalTensor<T> x1Gm, x2Gm, gammaGm, betaGm, xGm, biasGm, scalesGm, offsetsGm;

  GlobalTensor<float> workspaceGm;
  GlobalTensor<int8_t> yGm;

  uint32_t numCore;
  uint32_t numFirstDim;
  uint32_t numLastDim;
  uint32_t rowStep;
  uint32_t rowWork;
  uint32_t gmOffset_;
  uint32_t rowTail_;
  uint32_t nlFirstDimPerCore;
  uint32_t lFirstDimPerCore;
  float eps;
  float aveNum;
  bool lastDimPad = false;
  bool hasOffset = false;
  size_t numLastDimAligned;
};

#endif  // __QUANTIZE_ADD_LAYER_NORM_SINGLE_ROW_KERNEL_H_