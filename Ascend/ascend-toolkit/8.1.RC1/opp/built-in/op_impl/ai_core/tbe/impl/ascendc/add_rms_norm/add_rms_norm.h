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
 * \file add_rms_norm.h
 * \brief
 */
#ifndef _ADD_RMS_NORM_H_
#define _ADD_RMS_NORM_H_
#include "../rms_norm/rms_norm_base.h"

using namespace AscendC;

template <typename T>
class KernelAddRmsNorm {
 public:
  __aicore__ inline KernelAddRmsNorm(TPipe *pipe) {Ppipe = pipe;
  }
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x,
                              const AddRMSNormTilingData* tiling) {
    ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
    this->numRow = tiling->num_row;
    this->numCol = tiling->num_col;
    this->blockFactor = tiling->block_factor;
    this->rowFactor = tiling->row_factor;
    this->ubFactor = tiling->ub_factor;
    this->epsilon = tiling->epsilon;
    this->avgFactor = (numCol != 0) ? (float)1.0 / numCol : 0;

    blockIdx_ = GetBlockIdx();
    if (blockIdx_ < GetBlockNum() - 1) {
      this->rowWork = blockFactor;
    } else if (blockIdx_ == GetBlockNum() - 1) {
      this->rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
    }
    // get start index for current core, core parallel
    x1Gm.SetGlobalBuffer((__gm__ T*)x1 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
    x2Gm.SetGlobalBuffer((__gm__ T*)x2 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
    gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
    yGm.SetGlobalBuffer((__gm__ T*)y + blockIdx_ * blockFactor * numCol, rowWork * numCol);
    rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx_ * blockFactor, blockFactor);
    xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * blockFactor * numCol, rowWork * numCol);

    // pipe alloc memory to queue, the unit is Bytes
    Ppipe->InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(T));
    Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
    Ppipe->InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
    Ppipe->InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));

    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
    }
    Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
    Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
  }

  __aicore__ inline void Process() {
    CopyInGamma();
    LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();

    uint32_t i_o_max = CeilDiv(rowWork, rowFactor);
    uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;

    for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
      SubProcess(i_o, rowFactor, gammaLocal);
    }
    SubProcess(i_o_max - 1, row_tail, gammaLocal);
    inQueueGamma.FreeTensor(gammaLocal);
  }

  __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, LocalTensor<T>& gammaLocal) {
    LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
    for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
      uint32_t gm_bias = (i_o * rowFactor + i_i) * numCol;
      CopyIn(gm_bias);
      Compute(i_i, gammaLocal, rstdLocal);
      CopyOutY(gm_bias);
    }
    outQueueRstd.EnQue<float>(rstdLocal);
    CopyOutRstd(i_o, calc_row_num);
  }

 private:
  __aicore__ inline void CopyIn(uint32_t gm_bias) {
    LocalTensor<T> x1Local_in = inQueueX.AllocTensor<T>();
    LocalTensor<T> x2Local = sqxBuf.Get<T>();
    LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();

    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      x2Local = x2Local[ubFactor];
    }

    DataCopyCustom<T>(x1Local_in, x1Gm[gm_bias], numCol);
    DataCopyCustom<T>(x2Local, x2Gm[gm_bias], numCol);
    inQueueX.EnQue(x1Local_in);
    auto x1Local = inQueueX.DeQue<T>();

    if constexpr (is_same<T, half>::value) {
      LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
      Add(xLocal, x1Local, x2Local, numCol);
      pipe_barrier(PIPE_V);
      Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, numCol);
      pipe_barrier(PIPE_V);
    } else if constexpr (is_same<T, bfloat16_t>::value) {
      LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
      LocalTensor<float> x2_fp32 = sqxBuf.Get<float>();
      Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, numCol);
      Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, numCol);
      pipe_barrier(PIPE_V);
      Add(x1_fp32, x1_fp32, x2_fp32, numCol);
      pipe_barrier(PIPE_V);
      Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, numCol);
      pipe_barrier(PIPE_V);
    } else {
      Add(x1Local, x1Local, x2Local, numCol);
      pipe_barrier(PIPE_V);
      Adds(xLocal, x1Local, (float)0, numCol);
    }
    inQueueX.FreeTensor(x1Local);

    // CopyOut x1 + x2
    outQueueY.EnQue(xLocal);
    auto x_out = outQueueY.DeQue<T>();
    DataCopyCustom<T>(xGm[gm_bias], x_out, numCol);
    outQueueY.FreeTensor(x_out);
  }

  __aicore__ inline void CopyInGamma() {
    LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
    DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
    inQueueGamma.EnQue(gammaLocal);
  }

  __aicore__ inline void Compute(uint32_t inner_progress, LocalTensor<float> gammaLocal, LocalTensor<float> rstdLocal) {
    LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
    Mul(sqx, xLocal, xLocal, numCol);
    pipe_barrier(PIPE_V);

    Muls(sqx, sqx, avgFactor, numCol);
    pipe_barrier(PIPE_V);

    ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
    pipe_barrier(PIPE_V);
    Adds(sqx, sqx, epsilon, 1);
    pipe_barrier(PIPE_V);

    Sqrt(sqx, sqx, 1);
    Duplicate(reduce_buf_local, ONE, 1);
    pipe_barrier(PIPE_V);
    Div(sqx, reduce_buf_local, sqx, 1);
    pipe_barrier(PIPE_V);
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = sqx.GetValue(0);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    rstdLocal.SetValue(inner_progress, rstd_value);
    pipe_barrier(PIPE_V);
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    Muls(yLocal, xLocal, rstd_value, numCol);
    inQueueX.FreeTensor(xLocal);
    pipe_barrier(PIPE_V);
    Mul(yLocal, gammaLocal, yLocal, numCol);
    pipe_barrier(PIPE_V);
    outQueueY.EnQue<float>(yLocal);
  }

  __aicore__ inline void Compute(uint32_t inner_progress, LocalTensor<bfloat16_t> gammaLocal,
                                 LocalTensor<float> rstdLocal) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

    Mul(sqx, x_fp32, x_fp32, numCol);
    pipe_barrier(PIPE_V);

    Muls(sqx, sqx, avgFactor, numCol);
    pipe_barrier(PIPE_V);
    ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
    pipe_barrier(PIPE_V);

    Adds(sqx, sqx, epsilon, 1);
    pipe_barrier(PIPE_V);

    Sqrt(sqx, sqx, 1);
    Duplicate(reduce_buf_local, ONE, 1);
    pipe_barrier(PIPE_V);
    Div(sqx, reduce_buf_local, sqx, 1);
    pipe_barrier(PIPE_V);
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = sqx.GetValue(0);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    rstdLocal.SetValue(inner_progress, rstd_value);
    pipe_barrier(PIPE_V);
    Muls(x_fp32, x_fp32, rstd_value, numCol);
    pipe_barrier(PIPE_V);
    LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
    Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
    pipe_barrier(PIPE_V);
    Cast(x_fp32, yLocal, RoundMode::CAST_NONE, numCol);
    pipe_barrier(PIPE_V);
    Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol);  // gamma_fp32 reuse sqx
    pipe_barrier(PIPE_V);
    Mul(x_fp32, x_fp32, sqx, numCol);
    pipe_barrier(PIPE_V);
    Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
    pipe_barrier(PIPE_V);

    event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    set_flag(PIPE_V, PIPE_MTE2, event_v_mte);
    wait_flag(PIPE_V, PIPE_MTE2, event_v_mte);

    outQueueY.EnQue<bfloat16_t>(yLocal);
  }

  __aicore__ inline void Compute(uint32_t inner_progress, LocalTensor<half> gammaLocal, LocalTensor<float> rstdLocal) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

    Mul(sqx, x_fp32, x_fp32, numCol);
    pipe_barrier(PIPE_V);

    Muls(sqx, sqx, avgFactor, numCol);
    pipe_barrier(PIPE_V);

    ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
    pipe_barrier(PIPE_V);

    Adds(sqx, sqx, epsilon, 1);
    pipe_barrier(PIPE_V);

    Sqrt(sqx, sqx, 1);
    Duplicate(reduce_buf_local, ONE, 1);
    pipe_barrier(PIPE_V);
    Div(sqx, reduce_buf_local, sqx, 1);
    pipe_barrier(PIPE_V);
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = sqx.GetValue(0);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    rstdLocal.SetValue(inner_progress, rstd_value);
    pipe_barrier(PIPE_V);
    Muls(x_fp32, x_fp32, rstd_value, numCol);
    pipe_barrier(PIPE_V);
    LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
    Cast(yLocal, x_fp32, RoundMode::CAST_NONE, numCol);

    event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    set_flag(PIPE_V, PIPE_MTE2, event_v_mte);
    wait_flag(PIPE_V, PIPE_MTE2, event_v_mte);

    pipe_barrier(PIPE_V);
    Mul(yLocal, gammaLocal, yLocal, numCol);
    pipe_barrier(PIPE_V);
    outQueueY.EnQue<half>(yLocal);
  }

  __aicore__ inline void CopyOutY(uint32_t progress) {
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopyCustom<T>(yGm[progress], yLocal, numCol);
    outQueueY.FreeTensor(yLocal);
  }

  __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num) {
    LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if __CCE_AICORE__ == 220
    DataCopyCustom<float>(rstdGm[outer_progress * rowFactor], rstdLocal, num);
#endif
    outQueueRstd.FreeTensor(rstdLocal);
  }

 private:
  TPipe *Ppipe = nullptr;
  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueGamma;
  // create queues for output, in this case depth is equal to buffer num
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY, outQueueRstd;

  TBuf<TPosition::VECCALC> xFp32Buf;
  TBuf<TPosition::VECCALC> sqxBuf;
  TBuf<TPosition::VECCALC> reduceFp32Buf;
  GlobalTensor<T> x1Gm;
  GlobalTensor<T> x2Gm;
  GlobalTensor<T> gammaGm;
  GlobalTensor<T> yGm;
  GlobalTensor<float> rstdGm;
  GlobalTensor<T> xGm;

  uint32_t numRow;
  uint32_t numCol;
  uint32_t blockFactor;  // number of calculations rows on each core
  uint32_t rowFactor;
  uint32_t ubFactor;
  float epsilon;
  float avgFactor;
  int32_t blockIdx_;
  uint32_t rowWork = 1;
};
#endif  // ADD_RMS_NORM_H_