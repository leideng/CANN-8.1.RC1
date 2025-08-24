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
 * \file dua_quantize_add_layer_norm_single_row_kernel.h
 * \brief
 */
#ifndef __DUA_QUANTIZE_ADD_LAYER_NORM_SINGLE_ROW_KERNEL_H_
#define __DUA_QUANTIZE_ADD_LAYER_NORM_SINGLE_ROW_KERNEL_H_

#include "dua_quantize_add_layer_norm_base.h"

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelDuaQuantizeAddLayerNormSingleRow {
#define IS_ADDITIONAL_OUTPUT_ENABLE (1 == 1)
#define IS_MODE_MUL_SCALE ((TILING_KEY % 10) == 1)
 public:
  __aicore__ inline KernelDuaQuantizeAddLayerNormSingleRow(TPipe* pipe) {
    Ppipe = pipe;
  }
  __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y) {
    if (y > 0) {
      return (x + y - 1) / y;
    }
    return 0;
  }
  __aicore__ inline uint32_t ROUND_UP32(uint32_t x) {
    return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
  }
  __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y) {
    return x < y ? x : y;
  }
  __aicore__ inline uint32_t MAX(uint32_t x, uint32_t y) {
    return x > y ? x : y;
  }
  __aicore__ inline void Init(__gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* gamma, __gm__ uint8_t* beta,
                              __gm__ uint8_t* bias, __gm__ uint8_t* scales1, __gm__ uint8_t* scales2,
                              __gm__ uint8_t* zeroPoints1, __gm__ uint8_t* zeroPoints2, __gm__ uint8_t* y1,
                              __gm__ uint8_t* y2, __gm__ uint8_t* x, __gm__ uint8_t* workspace,
                              const DuaQuantizeAddLayerNormTilingData* tiling) {
    numCore = tiling->numCore;
    numLastDim = tiling->numLastDim;
    numFirstDim = tiling->numRow;
    nlFirstDimPerCore = tiling->nlFirstDimPerCore;
    lFirstDimPerCore = tiling->lFirstDimPerCore;
    firstDimPerTime = tiling->firstDimPerTime;
    lastDimPerTime = tiling->lastDimPerTime;
    aveNum = tiling->aveNum;
    eps = tiling->eps;
    colMoveCnt = tiling->colMoveCnt;
    colTail = tiling->colTail;
    isZeroPoint1Exist = tiling->isZeroPoint1Exist;
    isZeroPoint2Exist = tiling->isZeroPoint2Exist;
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
    scales1Gm.SetGlobalBuffer((__gm__ T*)scales1);
    scales2Gm.SetGlobalBuffer((__gm__ T*)scales2);
    if (isZeroPoint1Exist == 1) {
      zeroPoints1Gm.SetGlobalBuffer((__gm__ T*)zeroPoints1);
    }
    if (isZeroPoint2Exist == 1) {
      zeroPoints2Gm.SetGlobalBuffer((__gm__ T*)zeroPoints2);
    }
    y1Gm.SetGlobalBuffer((__gm__ int8_t*)(y1) + block_idx * gmOffset_);
    y2Gm.SetGlobalBuffer((__gm__ int8_t*)(y2) + block_idx * gmOffset_);
    xGm.SetGlobalBuffer((__gm__ T*)(x) + block_idx * gmOffset_);

    // single row
    Ppipe->InitBuffer(rowInQue, BUFFER_NUM, numLastDimAligned * sizeof(T));             // 2
    Ppipe->InitBuffer(rowOutQue, BUFFER_NUM, numLastDimAligned * sizeof(T));            // 2
    Ppipe->InitBuffer(quantizeOutQue, BUFFER_NUM, numLastDimAligned * sizeof(int8_t));  // 1
    Ppipe->InitBuffer(tmpBuf, 3 * numLastDimAligned * sizeof(float));                   // 12
  }
  __aicore__ inline void Process() {
    uint32_t gm_offset = 0;
    for (uint32_t row_idx = 0; row_idx < rowWork; ++row_idx) {
      CopyInAddSingleRow(gm_offset);
      precision_compute_single_row(gm_offset);
      gm_offset += numLastDim;
    }
  }

 private:
  __aicore__ inline void CopyInAddSingleRow(uint32_t gm_offset) {
    LocalTensor<float> tmpTensors = tmpBuf.Get<float>();

    LocalTensor<T> x1x2Fp32 = tmpTensors.ReinterpretCast<T>()[0];
    LocalTensor<T> x1In = x1x2Fp32[0];
    LocalTensor<T> x2In = x1x2Fp32[numLastDimAligned];

    LocalTensor<T> biasIn = rowInQue.template AllocTensor<T>();

    DataCopyEx(x1In, x1Gm[gm_offset], numLastDim);
    DataCopyEx(x2In, x2Gm[gm_offset], numLastDim);
    DataCopyEx(biasIn, biasGm, numLastDim);

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
      LocalTensor<float> xTensor = tmpTensors[0];
      LocalTensor<float> yTensor = tmpTensors[numLastDimAligned];
      LocalTensor<float> zTensor = tmpTensors[numLastDimAligned * 2];

      pipe_barrier(PIPE_V);
      Cast(yTensor, x1In, RoundMode::CAST_NONE, numLastDim);
      Cast(zTensor, x2In, RoundMode::CAST_NONE, numLastDim);
      pipe_barrier(PIPE_V);
      Add(yTensor, yTensor, zTensor, numLastDim);
      Cast(xTensor, biasTensor, RoundMode::CAST_NONE, numLastDim);
      pipe_barrier(PIPE_V);
      Add(xTensor, xTensor, yTensor, numLastDim);
      auto xOutTensor = rowOutQue.template AllocTensor<T>();
      pipe_barrier(PIPE_V);
      Cast(xOutTensor, xTensor, RoundMode::CAST_RINT, numLastDim);
      rowOutQue.template EnQue<T>(xOutTensor);
      auto xOut = rowOutQue.template DeQue<T>();
      DataCopyEx(xGm[gm_offset], xOut, numLastDim);
      rowOutQue.FreeTensor(xOut);
    }
    pipe_barrier(PIPE_V);
    rowInQue.FreeTensor(biasTensor);
  }

  __aicore__ inline void precision_compute_single_row(uint32_t gm_offset) {
    LocalTensor<float> tmpTensors = tmpBuf.Get<float>();

    LocalTensor<float> xTensor = tmpTensors[0];
    LocalTensor<float> tmpTensor = tmpTensors[numLastDimAligned];
    LocalTensor<float> constsTensor = tmpTensors[numLastDimAligned * 2];

    // CopyIn workspace beta gamma
    LocalTensor<T> scales01In = rowInQue.template AllocTensor<T>();
    LocalTensor<T> gammaLocal = constsTensor.ReinterpretCast<T>()[0];
    LocalTensor<T> betaLocal = constsTensor.ReinterpretCast<T>()[numLastDimAligned];

    DataCopyEx(gammaLocal, gammaGm, numLastDim);
    DataCopyEx(betaLocal, betaGm, numLastDim);
    DataCopyEx(scales01In, scales1Gm, numLastDim);

    Muls(tmpTensor, xTensor, aveNum, numLastDim);                 // x / N
    pipe_barrier(PIPE_V);
    float ave_local_temp = ReduceSumFP32(tmpTensor, numLastDim);  // E(X)
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    Adds(xTensor, xTensor, -1 * ave_local_temp, numLastDim);  // x - E(X)
    pipe_barrier(PIPE_V);
    Mul(tmpTensor, xTensor, xTensor, numLastDim);  // (x - E(X)) ** 2
    pipe_barrier(PIPE_V);
    Muls(tmpTensor, tmpTensor, aveNum, numLastDim);  //  ((x - E(X)) ** 2) / N
    pipe_barrier(PIPE_V);
    float var_local_temp = ReduceSumFP32(tmpTensor, numLastDim);  // Var(x)
    pipe_barrier(PIPE_V);
    float rstd_local_temp = 1 / sqrt(var_local_temp + eps);  // rstd = 1 / Sqrt(Var(x) + eps)
    event_t event_s_v_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v_1);
    wait_flag(PIPE_S, PIPE_V, event_s_v_1);
    Muls(xTensor, xTensor, rstd_local_temp, numLastDim);  // (x - E(X)) * rstd

    rowInQue.EnQue(scales01In);
    auto scales01Tensor = rowInQue.template DeQue<T>();

    Cast(tmpTensor, gammaLocal, RoundMode::CAST_NONE, numLastDim);  // tmpTensor <-- gammaFp32
    pipe_barrier(PIPE_V);
    Mul(xTensor, xTensor, tmpTensor, numLastDim);                     // xTensor <-- (x - E(X)) * rstd * gamma
    Cast(constsTensor, betaLocal, RoundMode::CAST_NONE, numLastDim);  // constsTensor <-- betaFp32
    pipe_barrier(PIPE_V);
    Cast(tmpTensor, scales01Tensor, RoundMode::CAST_NONE, numLastDim);  // tmpTensor <-- scales01Tensor
    rowInQue.FreeTensor(scales01Tensor);

    LocalTensor<T> offsets01In;
    if (isZeroPoint1Exist == 1) {
      offsets01In = rowInQue.template AllocTensor<T>();
      DataCopyEx(offsets01In, zeroPoints1Gm, numLastDim);
    }

    Add(xTensor, xTensor, constsTensor, numLastDim);  // xTensor <-- (x - E(X)) * rstd * gamma + beta
    pipe_barrier(PIPE_V);

    // small operator action
    Cast(constsTensor.ReinterpretCast<T>(), xTensor, RoundMode::CAST_RINT, numLastDim);
    pipe_barrier(PIPE_V);
    Cast(xTensor, constsTensor.ReinterpretCast<T>(), RoundMode::CAST_NONE, numLastDim);
    pipe_barrier(PIPE_V);

    if constexpr (IS_MODE_MUL_SCALE) {
      Mul(constsTensor, xTensor, tmpTensor, numLastDim);  // constsTensor <-- ((x - E(X)) * rstd * gamma + beta) * scales
    } else {
      Div(constsTensor, xTensor, tmpTensor, numLastDim);  // constsTensor <-- ((x - E(X)) * rstd * gamma + beta) / scales
    }
    pipe_barrier(PIPE_V);

    if (isZeroPoint1Exist == 1) {
      rowInQue.EnQue(offsets01In);
      auto offsets01Local = rowInQue.template DeQue<T>();
      Cast(tmpTensor, offsets01Local, RoundMode::CAST_NONE, numLastDim);
      pipe_barrier(PIPE_V);
      rowInQue.FreeTensor(offsets01Local);
      Add(constsTensor, constsTensor, tmpTensor, numLastDim);
      pipe_barrier(PIPE_V);
    }

    LocalTensor<int8_t> y1Local = quantizeOutQue.template AllocTensor<int8_t>();
    Cast(constsTensor.ReinterpretCast<int32_t>(), constsTensor, RoundMode::CAST_RINT, numLastDim);
    pipe_barrier(PIPE_V);
    SetDeqScale((half)1.000000e+00f);
    pipe_barrier(PIPE_V);
    Cast(constsTensor.ReinterpretCast<half>(), constsTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
         numLastDim);
    pipe_barrier(PIPE_V);
    Cast(y1Local, constsTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, numLastDim);
    quantizeOutQue.EnQue(y1Local);
    auto y1Out = quantizeOutQue.template DeQue<int8_t>();
    DataCopyEx(y1Gm[gm_offset], y1Out, numLastDim);
    quantizeOutQue.FreeTensor(y1Out);

    // quantize2
    LocalTensor<T> scales2In = rowInQue.template AllocTensor<T>();
    DataCopyEx(scales2In, scales2Gm, numLastDim);
    LocalTensor<T> offsets2Tensor;
    if (isZeroPoint2Exist == 1) {
      offsets2Tensor = rowOutQue.template AllocTensor<T>();
      event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
      set_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
      DataCopyEx(offsets2Tensor, zeroPoints2Gm, numLastDim);
    }
    rowInQue.EnQue(scales2In);
    auto scales2Tensor = rowInQue.template DeQue<T>();

    Cast(constsTensor, scales2Tensor, RoundMode::CAST_NONE, numLastDim);
    if (isZeroPoint2Exist == 1) {
      Cast(tmpTensor, offsets2Tensor, RoundMode::CAST_NONE, numLastDim);
      rowOutQue.FreeTensor(offsets2Tensor);
    }
    pipe_barrier(PIPE_V);
    rowInQue.FreeTensor(scales2Tensor);

    if constexpr (IS_MODE_MUL_SCALE) {
      Mul(constsTensor, xTensor, constsTensor, numLastDim);  // ((x - E(X)) * rstd * gamma + beta) * scales
    } else {
      Div(constsTensor, xTensor, constsTensor, numLastDim);  // ((x - E(X)) * rstd * gamma + beta) / scales
    }
    event_t event_v_mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    set_flag(PIPE_V, PIPE_MTE2, event_v_mte2);
    wait_flag(PIPE_V, PIPE_MTE2, event_v_mte2);

    pipe_barrier(PIPE_V);
    if (isZeroPoint2Exist == 1) {
      Add(constsTensor, constsTensor, tmpTensor, numLastDim);
      pipe_barrier(PIPE_V);
    }

    LocalTensor<int8_t> y2Local = quantizeOutQue.template AllocTensor<int8_t>();
    Cast(constsTensor.ReinterpretCast<int32_t>(), constsTensor, RoundMode::CAST_RINT, numLastDim);
    pipe_barrier(PIPE_V);
    SetDeqScale((half)1.000000e+00f);
    pipe_barrier(PIPE_V);
    Cast(constsTensor.ReinterpretCast<half>(), constsTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
         numLastDim);
    pipe_barrier(PIPE_V);
    Cast(y2Local, constsTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, numLastDim);
    quantizeOutQue.EnQue(y2Local);
    auto y2Out = quantizeOutQue.template DeQue<int8_t>();
    DataCopyEx(y2Gm[gm_offset], y2Out, numLastDim);
    quantizeOutQue.FreeTensor(y2Out);
  }

 private:
  TPipe* Ppipe = nullptr;
  TQue<QuePosition::VECIN, BUFFER_NUM> rowInQue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> rowOutQue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> quantizeOutQue;

  TBuf<TPosition::VECCALC> tmpBuf;
  GlobalTensor<T> x1Gm, x2Gm, gammaGm, betaGm, xGm, biasGm, scales1Gm, scales2Gm, zeroPoints1Gm, zeroPoints2Gm;
  GlobalTensor<float> workspaceGm;
  GlobalTensor<int8_t> y1Gm, y2Gm;

  uint32_t numCore;
  uint32_t numFirstDim;
  uint32_t numLastDim;
  uint32_t rowStep;
  uint32_t rowWork;
  uint32_t gmOffset_;
  uint32_t rowTail_;
  uint32_t colTail;
  uint32_t colMoveCnt;
  uint32_t firstDimPerTime;
  uint32_t lastDimPerTime;
  uint32_t nlFirstDimPerCore;
  uint32_t lFirstDimPerCore;
  float eps;
  float aveNum;
  bool lastDimPad = false;
  size_t numLastDimAligned;
  uint32_t isZeroPoint1Exist;
  uint32_t isZeroPoint2Exist;
};

#endif  // __DUA_QUANTIZE_ADD_LAYER_NORM_SINGLE_ROW_KERNEL_H_