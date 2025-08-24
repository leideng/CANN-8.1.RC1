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
 * \file rms_norm_grad_whole_reduce_d.h
 * \brief
 */
#ifndef RMS_NORM_GRAD_WHOLE_REDUCE_D_H
#define RMS_NORM_GRAD_WHOLE_REDUCE_D_H

#include "rms_norm_grad_common.h"
template <typename T>
class RmsNormGradWholeReduceD {
 public:
  __aicore__ inline RmsNormGradWholeReduceD(){};
  __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma,
                              const RmsNormGradTilingData* tiling);
  __aicore__ inline void Process();

 protected:
  TPipe pipe;
  GlobalTensor<T> dyGm, gammaGm, dxGm, xGm;
  GlobalTensor<float> dgammaGm, rstdGm;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueDY, inQueX, inQueRstd, inQueGamma;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueDX, outQueDgamma;
  TBuf<TPosition::VECCALC> ndBufFp32Buf1, ndBufFp32Buf2, ndBufFp32Buf3;
  TBuf<TPosition::VECCALC> dFp32Buf;
  TBuf<TPosition::VECCALC> nFp32Buf;

  // tilingdata
  uint32_t rowVal;
  uint32_t colVal;
  float avgFactor;
  uint32_t dataType;
  uint32_t blockFactor;
  uint32_t ubFactor;
  uint32_t coreCalcNum;
  uint32_t coreCalcTail;
  uint32_t blockDim;
  uint32_t ubCalcNum;
  uint32_t ubCalcTail;
  uint32_t ubCalcLoop;
  uint32_t ubCalcTailNum;
  uint32_t ubCalcTailTail;
  uint32_t ubCalcTailLoop;
  uint32_t fixedOutput{0};

  uint32_t blockIdx;
  uint32_t alignLen;
  uint32_t coreOffset;
  uint32_t ubFactorAlign;
  uint32_t rstdLen;
  uint32_t bufferLenSize;
  uint32_t bufferNum{1};
  uint32_t ubTailAlign;
  uint64_t fp32Mask = ONE_REPEAT_BYTE_SIZE / sizeof(float);

  __aicore__ inline void InitData(const RmsNormGradTilingData* tiling);
  __aicore__ inline void CopyIn(uint32_t nIdx, uint32_t dIdx, uint32_t calcUnit);
  __aicore__ inline void CopyOut(uint32_t nIdx, uint32_t dIdx, uint32_t calcUnit);
  __aicore__ inline void CopyGammaIn(uint32_t dIdx, uint32_t calcLen);
  __aicore__ inline void CopyDgammaOut(uint32_t dIdx, uint32_t calcLen);
  __aicore__ inline void CopyDgammaOutInOrder(uint32_t dIdx, uint32_t calcLen);
  __aicore__ inline void ComputeDgammaMain(uint32_t loopLen);
  __aicore__ inline void ComputeDgamma(uint32_t i, uint32_t j, uint32_t calcLen, LocalTensor<T>& gammaUb,
                                       LocalTensor<float>& dgamma);
  __aicore__ inline void ComputeMain(uint32_t calcLen, LocalTensor<T>& gammaUb, float dySumVal);
  __aicore__ inline void ComputeDySum(uint32_t i, uint32_t j, uint32_t calcLen, uint32_t calcLenAlign,
                                      LocalTensor<float>& dySum);
  __aicore__ inline void ProcessMain(uint32_t loopLen);
};

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceD<T>::Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx,
                                                        GM_ADDR dgamma, const RmsNormGradTilingData* tiling) {
  InitData(tiling);

  blockIdx = GetBlockIdx();
  if (blockIdx < blockDim - 1) {
    coreOffset = blockFactor;
  } else {
    coreOffset = coreCalcTail > 0 ? coreCalcTail : blockFactor;
  }
  dyGm.SetGlobalBuffer((__gm__ T*)dy + blockIdx * blockFactor * colVal, coreOffset * colVal);
  xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx * blockFactor * colVal, coreOffset * colVal);
  rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx * blockFactor, coreOffset);
  gammaGm.SetGlobalBuffer((__gm__ T*)gamma, colVal);
  dxGm.SetGlobalBuffer((__gm__ T*)dx + blockIdx * blockFactor * colVal, coreOffset * colVal);
  dgammaGm.SetGlobalBuffer((__gm__ float*)dgamma, colVal);

  ubFactorAlign = (ubFactor + alignLen - 1) / alignLen * alignLen;
  rstdLen = alignLen;
  bufferLenSize = ubFactorAlign * sizeof(T);
  pipe.InitBuffer(inQueDY, bufferNum, bufferLenSize);
  pipe.InitBuffer(inQueX, bufferNum, bufferLenSize);
  pipe.InitBuffer(inQueRstd, bufferNum, rstdLen * sizeof(float));
  pipe.InitBuffer(inQueGamma, bufferNum, bufferLenSize);
  pipe.InitBuffer(outQueDX, bufferNum, bufferLenSize);
  pipe.InitBuffer(outQueDgamma, bufferNum, ubFactorAlign * sizeof(float));

  uint32_t tmpBufferSize = ubFactorAlign * sizeof(float);
  pipe.InitBuffer(ndBufFp32Buf1, tmpBufferSize);
  pipe.InitBuffer(nFp32Buf, colVal * sizeof(float));
  if constexpr (is_same<T, half>::value) {
    pipe.InitBuffer(ndBufFp32Buf2, tmpBufferSize);
    pipe.InitBuffer(ndBufFp32Buf3, tmpBufferSize);
    pipe.InitBuffer(dFp32Buf, tmpBufferSize);
  }

  if (blockIdx == 0) {
    LocalTensor<float> initGmZeros = nFp32Buf.Get<float>();
    Duplicate(initGmZeros, 0.0f, colVal);
    pipe_barrier(PIPE_V);
    DataCopy(dgammaGm, initGmZeros, colVal);
    pipe_barrier(PIPE_V);
    fixed_output_sync[0] = 0;
  }
  SyncAll();
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceD<T>::InitData(const RmsNormGradTilingData* tiling) {
  rowVal = tiling->row;
  colVal = tiling->col;
  avgFactor = tiling->avg_factor;
  dataType = tiling->data_type;
  blockFactor = tiling->block_factor;
  ubFactor = tiling->ub_factor;
  coreCalcNum = tiling->core_calc_num;
  coreCalcTail = tiling->core_calc_tail;
  blockDim = tiling->block_dim;
  ubCalcNum = tiling->ub_calc_num;
  ubCalcTail = tiling->ub_calc_tail;
  ubCalcLoop = tiling->ub_calc_loop;
  ubCalcTailNum = tiling->ub_calc_tail_num;
  ubCalcTailTail = tiling->ub_calc_tail_tail;
  ubCalcTailLoop = tiling->ub_calc_tail_loop;
  fixedOutput = tiling->fixed_output;

  if constexpr (is_same<T, half>::value) {
    alignLen = ALIGN_16;
    bufferNum = BUFFER_NUM;
  } else if constexpr (is_same<T, float>::value) {
    alignLen = ALIGN_32;
    bufferNum = BUFFER_NUM;
  }
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceD<T>::CopyIn(uint32_t nIdx, uint32_t dIdx, uint32_t calcUnit) {
  uint32_t gmOffset = nIdx * colVal + dIdx * ubFactor;
  LocalTensor<float> rstd = inQueRstd.AllocTensor<float>();
  DataCopy(rstd, rstdGm[nIdx], alignLen);
  inQueRstd.EnQue(rstd);
  LocalTensor<T> x = inQueX.AllocTensor<T>();
  DataCopy(x, xGm[gmOffset], calcUnit);
  inQueX.EnQue(x);
  LocalTensor<T> dy = inQueDY.AllocTensor<T>();
  DataCopy(dy, dyGm[gmOffset], calcUnit);
  inQueDY.EnQue(dy);
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceD<T>::CopyOut(uint32_t nIdx, uint32_t dIdx, uint32_t calcUnit) {
  LocalTensor<T> dx = outQueDX.DeQue<T>();
  DataCopy(dxGm[nIdx * colVal + dIdx * ubFactor], dx, calcUnit);
  outQueDX.FreeTensor(dx);
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceD<T>::CopyGammaIn(uint32_t dIdx, uint32_t calcLen) {
  LocalTensor<T> gammaUb = inQueGamma.AllocTensor<T>();
  DataCopy(gammaUb, gammaGm[dIdx * ubFactor], calcLen);
  inQueGamma.EnQue(gammaUb);
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceD<T>::CopyDgammaOut(uint32_t dIdx, uint32_t calcLen) {
  LocalTensor<float> dgammaOut = outQueDgamma.DeQue<float>();
  SetAtomicAdd<float>();
  DataCopy(dgammaGm[dIdx * ubFactor], dgammaOut, calcLen);
  SetAtomicNone();
  outQueDgamma.FreeTensor(dgammaOut);
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceD<T>::CopyDgammaOutInOrder(uint32_t dIdx, uint32_t calcLen) {
  int32_t whileMax = 2147483647;
  for (int32_t count = 0; count < whileMax; count++) {
    if (fixed_output_sync[0] == blockIdx + dIdx * blockDim) {
      whileMax = 0;
    }
  }
  pipe_barrier(PIPE_ALL);
  LocalTensor<float> dgammaOut = outQueDgamma.DeQue<float>();
  SetAtomicAdd<float>();
  DataCopy(dgammaGm[dIdx * ubFactor], dgammaOut, calcLen);
  SetAtomicNone();
  outQueDgamma.FreeTensor(dgammaOut);
  fixed_output_sync[0] = fixed_output_sync[0] + 1;
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceD<T>::ComputeDgammaMain(uint32_t loopLen) {
  for (uint32_t j = 0; j < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); j++) {
    CopyGammaIn(j, ubFactor);
    LocalTensor<T> gammaUb = inQueGamma.DeQue<T>();
    LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
    Duplicate(dgamma, 0.0f, ubFactor);
    pipe_barrier(PIPE_V);
    for (uint32_t i = 0; i < loopLen; i++) {
      CopyIn(i, j, ubFactor);
      ComputeDgamma(i, j, ubFactor, gammaUb, dgamma);
    }
    inQueGamma.FreeTensor(gammaUb);
    outQueDgamma.EnQue(dgamma);
    if (fixedOutput == 1) {
      CopyDgammaOutInOrder(j, ubFactor);
    } else {
      CopyDgammaOut(j, ubFactor);
    }
  }
  if (ubCalcTail != 0) {
    CopyGammaIn(ubCalcLoop - 1, ubCalcTail);
    LocalTensor<T> gammaUb = inQueGamma.DeQue<T>();
    LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
    Duplicate(dgamma, 0.0f, ubCalcTail);
    pipe_barrier(PIPE_V);
    for (uint32_t i = 0; i < loopLen; i++) {
      CopyIn(i, ubCalcLoop - 1, ubCalcTail);
      ComputeDgamma(i, ubCalcLoop - 1, ubCalcTail, gammaUb, dgamma);
    }
    inQueGamma.FreeTensor(gammaUb);
    outQueDgamma.EnQue(dgamma);
    if (fixedOutput == 1) {
      CopyDgammaOutInOrder(ubCalcLoop - 1, ubCalcTail);
    } else {
      CopyDgammaOut(ubCalcLoop - 1, ubCalcTail);
    }
  }
}

template <>
__aicore__ inline void RmsNormGradWholeReduceD<half>::ComputeDgamma(uint32_t i, uint32_t j, uint32_t calcLen,
                                                                    LocalTensor<half>& gammaUb,
                                                                    LocalTensor<float>& dgamma) {
  LocalTensor<half> xUb = inQueX.DeQue<half>();
  LocalTensor<half> dyUb = inQueDY.DeQue<half>();
  LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
  LocalTensor<float> tmp32Buf2 = ndBufFp32Buf2.Get<float>();
  LocalTensor<float> tmp32Buf3 = ndBufFp32Buf3.Get<float>();
  // dy * (x * rstd)  -> sum
  event_t eventMteS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
  set_flag(PIPE_MTE2, PIPE_S, eventMteS);
  wait_flag(PIPE_MTE2, PIPE_S, eventMteS);
  float rstdValue = rstdUb.GetValue(0);
  event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  set_flag(PIPE_S, PIPE_V, eventSV);
  wait_flag(PIPE_S, PIPE_V, eventSV);
  Cast(tmp32Buf2, xUb, RoundMode::CAST_NONE, calcLen);
  pipe_barrier(PIPE_V);
  Muls(tmp32Buf2, tmp32Buf2, rstdValue, calcLen);
  pipe_barrier(PIPE_V);
  inQueRstd.FreeTensor(rstdUb);
  Cast(tmp32Buf3, dyUb, RoundMode::CAST_NONE, calcLen);
  pipe_barrier(PIPE_V);
  Mul(tmp32Buf2, tmp32Buf3, tmp32Buf2, calcLen);
  pipe_barrier(PIPE_V);
  inQueX.FreeTensor(xUb);
  inQueDY.FreeTensor(dyUb);
  Add(dgamma, dgamma, tmp32Buf2, calcLen);
  pipe_barrier(PIPE_V);
}

template <>
__aicore__ inline void RmsNormGradWholeReduceD<float>::ComputeDgamma(uint32_t i, uint32_t j, uint32_t calcLen,
                                                                     LocalTensor<float>& gammaUb,
                                                                     LocalTensor<float>& dgamma) {
  LocalTensor<float> xUb = inQueX.DeQue<float>();
  LocalTensor<float> dyUb = inQueDY.DeQue<float>();
  LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
  // dy * (x * rstd)  -> sum
  event_t eventMteS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
  set_flag(PIPE_MTE2, PIPE_S, eventMteS);
  wait_flag(PIPE_MTE2, PIPE_S, eventMteS);
  float rstdValue = rstdUb.GetValue(0);
  event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  set_flag(PIPE_S, PIPE_V, eventSV);
  wait_flag(PIPE_S, PIPE_V, eventSV);
  Muls(xUb, xUb, rstdValue, calcLen);
  pipe_barrier(PIPE_V);
  inQueRstd.FreeTensor(rstdUb);
  Mul(dyUb, dyUb, xUb, calcLen);
  pipe_barrier(PIPE_V);
  inQueX.FreeTensor(xUb);
  Add(dgamma, dgamma, dyUb, calcLen);
  pipe_barrier(PIPE_V);
  inQueDY.FreeTensor(dyUb);
}

template <>
__aicore__ inline void RmsNormGradWholeReduceD<half>::ComputeMain(uint32_t calcLen, LocalTensor<half>& gammaUb,
                                                                  float dySumVal) {
  LocalTensor<half> xUb = inQueX.DeQue<half>();
  LocalTensor<half> dyUb = inQueDY.DeQue<half>();
  LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
  LocalTensor<float> tmp32Buf1 = ndBufFp32Buf1.Get<float>();
  LocalTensor<float> tmp32Buf2 = ndBufFp32Buf2.Get<float>();
  LocalTensor<float> tmp32Buf3 = ndBufFp32Buf3.Get<float>();
  event_t eventMteS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
  set_flag(PIPE_MTE2, PIPE_S, eventMteS);
  wait_flag(PIPE_MTE2, PIPE_S, eventMteS);
  float rstdValue = rstdUb.GetValue(0);
  event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  set_flag(PIPE_S, PIPE_V, eventSV);
  wait_flag(PIPE_S, PIPE_V, eventSV);
  Cast(tmp32Buf2, xUb, RoundMode::CAST_NONE, calcLen);
  pipe_barrier(PIPE_V);
  inQueX.FreeTensor(xUb);
  inQueRstd.FreeTensor(rstdUb);
  Muls(tmp32Buf2, tmp32Buf2, rstdValue, calcLen);
  pipe_barrier(PIPE_V);
  Cast(tmp32Buf1, dyUb, RoundMode::CAST_NONE, calcLen);
  pipe_barrier(PIPE_V);
  inQueDY.FreeTensor(dyUb);
  Cast(tmp32Buf3, gammaUb, RoundMode::CAST_NONE, calcLen);
  pipe_barrier(PIPE_V);
  inQueGamma.FreeTensor(gammaUb);
  if (fixedOutput == 1) {
    Mul(tmp32Buf1, tmp32Buf1, tmp32Buf3, calcLen);
    pipe_barrier(PIPE_V);
  } else {
    LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
    Mul(dgamma, tmp32Buf1, tmp32Buf2, calcLen);
    pipe_barrier(PIPE_V);
    outQueDgamma.EnQue(dgamma);
    Mul(tmp32Buf1, tmp32Buf1, tmp32Buf3, calcLen);
    pipe_barrier(PIPE_V);
  }
  Muls(tmp32Buf2, tmp32Buf2, dySumVal, calcLen);
  pipe_barrier(PIPE_V);
  Sub(tmp32Buf1, tmp32Buf1, tmp32Buf2, calcLen);
  pipe_barrier(PIPE_V);
  Muls(tmp32Buf1, tmp32Buf1, rstdValue, calcLen);
  pipe_barrier(PIPE_V);
  LocalTensor<half> dxUb = outQueDX.AllocTensor<half>();
  Cast(dxUb, tmp32Buf1, RoundMode::CAST_NONE, calcLen);
  pipe_barrier(PIPE_V);
  outQueDX.EnQue(dxUb);
}

template <>
__aicore__ inline void RmsNormGradWholeReduceD<float>::ComputeMain(uint32_t calcLen, LocalTensor<float>& gammaUb,
                                                                   float dySumVal) {
  LocalTensor<float> xUb = inQueX.DeQue<float>();
  LocalTensor<float> dyUb = inQueDY.DeQue<float>();
  LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
  event_t eventMteS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
  set_flag(PIPE_MTE2, PIPE_S, eventMteS);
  wait_flag(PIPE_MTE2, PIPE_S, eventMteS);
  float rstdValue = rstdUb.GetValue(0);
  event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  set_flag(PIPE_S, PIPE_V, eventSV);
  wait_flag(PIPE_S, PIPE_V, eventSV);
  Muls(xUb, xUb, rstdValue, calcLen);
  pipe_barrier(PIPE_V);
  inQueRstd.FreeTensor(rstdUb);
  // grad_y = grad* gamma
  Mul(dyUb, dyUb, gammaUb, calcLen);
  Muls(xUb, xUb, dySumVal, calcLen);
  pipe_barrier(PIPE_V);
  inQueGamma.FreeTensor(gammaUb);
  Sub(dyUb, dyUb, xUb, calcLen);
  pipe_barrier(PIPE_V);
  inQueX.FreeTensor(xUb);
  LocalTensor<float> dxUb = outQueDX.AllocTensor<float>();
  Muls(dxUb, dyUb, rstdValue, calcLen);
  pipe_barrier(PIPE_V);
  inQueDY.FreeTensor(dyUb);
  outQueDX.EnQue(dxUb);
}

template <>
__aicore__ inline void RmsNormGradWholeReduceD<half>::ComputeDySum(uint32_t i, uint32_t j, uint32_t calcLen,
                                                                   uint32_t calcLenAlign, LocalTensor<float>& dySum) {
  CopyGammaIn(j, calcLen);
  CopyIn(i, j, calcLen);
  LocalTensor<half> gammaUb = inQueGamma.DeQue<half>();
  LocalTensor<half> xUb = inQueX.DeQue<half>();
  LocalTensor<half> dyUb = inQueDY.DeQue<half>();
  LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
  event_t eventMteS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
  set_flag(PIPE_MTE2, PIPE_S, eventMteS);
  wait_flag(PIPE_MTE2, PIPE_S, eventMteS);
  float rstd_value = rstdUb.GetValue(0);
  event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  set_flag(PIPE_S, PIPE_V, eventSV);
  wait_flag(PIPE_S, PIPE_V, eventSV);

  LocalTensor<float> dySumPart = ndBufFp32Buf1.Get<float>();
  LocalTensor<float> tmp32Buf = ndBufFp32Buf2.Get<float>();
  LocalTensor<float> tmp32Buf2 = ndBufFp32Buf3.Get<float>();

  Duplicate(dySumPart, 0.0f, ubTailAlign);
  // grad_y = dy * gamma
  Cast(tmp32Buf, dyUb, RoundMode::CAST_NONE, calcLen);
  pipe_barrier(PIPE_V);
  Cast(dySumPart, gammaUb, RoundMode::CAST_NONE, calcLen);
  pipe_barrier(PIPE_V);
  Mul(tmp32Buf, tmp32Buf, dySumPart, calcLen);
  Cast(tmp32Buf2, xUb, RoundMode::CAST_NONE, calcLen);
  pipe_barrier(PIPE_V);
  inQueGamma.FreeTensor(gammaUb);
  inQueRstd.FreeTensor(rstdUb);
  inQueX.FreeTensor(xUb);
  inQueDY.FreeTensor(dyUb);
  // sum(x * rstd * grad_y)
  Muls(tmp32Buf2, tmp32Buf2, rstd_value, calcLen);
  pipe_barrier(PIPE_V);
  Mul(dySumPart, tmp32Buf, tmp32Buf2, calcLen);
  pipe_barrier(PIPE_V);
  uint32_t colSplitNum = calcLen;
  uint32_t colSplitTailNum = 0;
  while (colSplitNum >= 32) {
    colSplitTailNum = colSplitNum % 16;
    if (colSplitTailNum != 0) {
      colSplitNum = colSplitNum - colSplitTailNum;
      Add(dySumPart, dySumPart, dySumPart[colSplitNum], colSplitTailNum);
      pipe_barrier(PIPE_V);
    }
    Add(dySumPart, dySumPart, dySumPart[colSplitNum / 2], colSplitNum / 2);
    pipe_barrier(PIPE_V);
    colSplitNum = colSplitNum / 2;
  }
  WholeReduceSum(dySumPart, dySumPart, colSplitNum, 1, 1, 1, 8);
  pipe_barrier(PIPE_V);
  Add(dySum, dySum, dySumPart, alignLen);
  pipe_barrier(PIPE_V);
}

template <>
__aicore__ inline void RmsNormGradWholeReduceD<float>::ComputeDySum(uint32_t i, uint32_t j, uint32_t calcLen,
                                                                    uint32_t calcLenAlign, LocalTensor<float>& dySum) {
  CopyGammaIn(j, calcLen);
  CopyIn(i, j, calcLen);
  LocalTensor<float> gammaUb = inQueGamma.DeQue<float>();
  LocalTensor<float> xUb = inQueX.DeQue<float>();
  LocalTensor<float> dyUb = inQueDY.DeQue<float>();
  LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
  event_t eventMteS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
  set_flag(PIPE_MTE2, PIPE_S, eventMteS);
  wait_flag(PIPE_MTE2, PIPE_S, eventMteS);
  float rstd_value = rstdUb.GetValue(0);
  event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  set_flag(PIPE_S, PIPE_V, eventSV);
  wait_flag(PIPE_S, PIPE_V, eventSV);
  LocalTensor<float> dySumPart = ndBufFp32Buf1.Get<float>();
  Duplicate(dySumPart, 0.0f, ubTailAlign);
  pipe_barrier(PIPE_V);
  // grad_y = dy * gamma
  Mul(dyUb, dyUb, gammaUb, calcLen);
  Muls(xUb, xUb, rstd_value, calcLen);
  pipe_barrier(PIPE_V);
  inQueGamma.FreeTensor(gammaUb);
  inQueRstd.FreeTensor(rstdUb);
  // sum(x * rstd * grad_y)
  Mul(dySumPart, dyUb, xUb, calcLen);
  pipe_barrier(PIPE_V);
  inQueX.FreeTensor(xUb);
  inQueDY.FreeTensor(dyUb);
  uint32_t colSplitNum = calcLen;
  uint32_t colSplitTailNum = 0;
  while (colSplitNum >= 32) {
    colSplitTailNum = colSplitNum % 16;
    if (colSplitTailNum != 0) {
      colSplitNum = colSplitNum - colSplitTailNum;
      Add(dySumPart, dySumPart, dySumPart[colSplitNum], colSplitTailNum);
      pipe_barrier(PIPE_V);
    }
    Add(dySumPart, dySumPart, dySumPart[colSplitNum / 2], colSplitNum / 2);
    pipe_barrier(PIPE_V);
    colSplitNum = colSplitNum / 2;
  }
  WholeReduceSum(dySumPart, dySumPart, colSplitNum, 1, 1, 1, 8);
  pipe_barrier(PIPE_V);
  Add(dySum, dySum, dySumPart, alignLen);
  pipe_barrier(PIPE_V);
}

template <>
__aicore__ inline void RmsNormGradWholeReduceD<half>::ProcessMain(uint32_t loopLen) {
  for (uint32_t i = 0; i < loopLen; i++) {
    // Calc mean firstly
    LocalTensor<float> dySum = nFp32Buf.Get<float>();
    Duplicate(dySum, 0.0f, alignLen);
    pipe_barrier(PIPE_V);
    for (uint32_t j = 0; j < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); j++) {
      ComputeDySum(i, j, ubFactor, ubFactor, dySum);
    }
    if (ubCalcTail != 0) {
      ubTailAlign = (ubCalcTail + alignLen - 1) / alignLen * alignLen;
      ComputeDySum(i, ubCalcLoop - 1, ubCalcTail, ubTailAlign, dySum);
    }
    Muls(dySum, dySum, avgFactor, alignLen);
    pipe_barrier(PIPE_V);
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, eventVS);
    wait_flag(PIPE_V, PIPE_S, eventVS);
    float dySumVal = dySum.GetValue(0);
    event_t eventSMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    set_flag(PIPE_S, PIPE_MTE2, eventSMte2);
    wait_flag(PIPE_S, PIPE_MTE2, eventSMte2);
    for (uint32_t j = 0; j < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); j++) {
      CopyIn(i, j, ubFactor);
      CopyGammaIn(j, ubFactor);
      LocalTensor<half> gammaUb = inQueGamma.DeQue<half>();
      ComputeMain(ubFactor, gammaUb, dySumVal);
      if (fixedOutput == 0) {
        CopyDgammaOut(j, ubFactor);
      }
      CopyOut(i, j, ubFactor);
    }
    if (ubCalcTail != 0) {
      CopyIn(i, ubCalcLoop - 1, ubCalcTail);
      CopyGammaIn(ubCalcLoop - 1, ubCalcTail);
      LocalTensor<half> gammaUb = inQueGamma.DeQue<half>();
      ComputeMain(ubCalcTail, gammaUb, dySumVal);
      if (fixedOutput == 0) {
        CopyDgammaOut(ubCalcLoop - 1, ubCalcTail);
      }
      CopyOut(i, ubCalcLoop - 1, ubCalcTail);
    }
  }
  if (fixedOutput == 1) {
    ComputeDgammaMain(loopLen);
  }
}

template <>
__aicore__ inline void RmsNormGradWholeReduceD<float>::ProcessMain(uint32_t loopLen) {
  for (uint32_t i = 0; i < loopLen; i++) {
    // Calc mean firstly
    LocalTensor<float> dySum = nFp32Buf.Get<float>();
    Duplicate(dySum, 0.0f, alignLen);
    pipe_barrier(PIPE_V);
    for (uint32_t j = 0; j < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); j++) {
      ComputeDySum(i, j, ubFactor, ubFactor, dySum);
    }
    if (ubCalcTail != 0) {
      ubTailAlign = (ubCalcTail + alignLen - 1) / alignLen * alignLen;
      ComputeDySum(i, ubCalcLoop - 1, ubCalcTail, ubTailAlign, dySum);
    }
    Muls(dySum, dySum, avgFactor, alignLen);
    pipe_barrier(PIPE_V);
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, eventVS);
    wait_flag(PIPE_V, PIPE_S, eventVS);
    float dySumVal = dySum.GetValue(0);
    event_t eventSMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    set_flag(PIPE_S, PIPE_MTE2, eventSMte2);
    wait_flag(PIPE_S, PIPE_MTE2, eventSMte2);
    for (uint32_t j = 0; j < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); j++) {
      CopyGammaIn(j, ubFactor);
      LocalTensor<float> gammaUb = inQueGamma.DeQue<float>();
      CopyIn(i, j, ubFactor);
      ComputeMain(ubFactor, gammaUb, dySumVal);
      CopyOut(i, j, ubFactor);
    }
    if (ubCalcTail != 0) {
      CopyGammaIn(ubCalcLoop - 1, ubCalcTail);
      LocalTensor<float> gammaUb = inQueGamma.DeQue<float>();
      CopyIn(i, ubCalcLoop - 1, ubCalcTail);
      ComputeMain(ubCalcTail, gammaUb, dySumVal);
      CopyOut(i, ubCalcLoop - 1, ubCalcTail);
    }
  }
  ComputeDgammaMain(loopLen);
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceD<T>::Process() {
  if (coreCalcTail == 0) {
    ProcessMain(blockFactor);
  } else {
    if (blockIdx < blockDim - 1) {
      ProcessMain(blockFactor);
    } else {
      ProcessMain(coreCalcTail);
    }
  }
}

#endif  // RMS_NORM_GRAD_WHOLE_REDUCE_D_H