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
 * \file add_rms_norm_split_d.h
 * \brief
 */
#ifndef _ADD_RMS_NORM_SPLIT_D_H_
#define _ADD_RMS_NORM_SPLIT_D_H_
#include "../rms_norm/rms_norm_base.h"

using namespace AscendC;

template <typename T>
class KernelAddRmsNormSplitD {
 public:
  __aicore__ inline KernelAddRmsNormSplitD(TPipe *pipe) {Ppipe = pipe;
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
    } else {
    }
    // get start index for current core, core parallel
    x1Gm.SetGlobalBuffer((__gm__ T*)x1 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
    x2Gm.SetGlobalBuffer((__gm__ T*)x2 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
    gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
    yGm.SetGlobalBuffer((__gm__ T*)y + blockIdx_ * blockFactor * numCol, rowWork * numCol);
    rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx_ * blockFactor, blockFactor);
    xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * blockFactor * numCol, rowWork * numCol);

    // pipe alloc memory to queue, the unit is Bytes.
    // We need 2 buffers here for both x1 and x2.
    Ppipe->InitBuffer(inQueueX, BUFFER_NUM, 2 * ubFactor * sizeof(T));
    Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
    Ppipe->InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
    Ppipe->InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));

    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
    }
    Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
    Ppipe->InitBuffer(sumBuf, rowFactor * NUM_PER_BLK_FP32 * sizeof(float));
    Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
  }

  __aicore__ inline void Process() {
    uint32_t i_o_max = CeilDiv(rowWork, rowFactor);
    uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;
    uint32_t j_max = CeilDiv(numCol, ubFactor);
    uint32_t col_tail = numCol - (j_max - 1) * ubFactor;
    for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
      SubProcess(i_o, rowFactor, j_max, col_tail);
    }
    SubProcess(i_o_max - 1, row_tail, j_max, col_tail);
  }

  __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, uint32_t j_max, uint32_t col_tail) {
    LocalTensor<float> sumLocal = sumBuf.Get<float>();

    LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
    Duplicate(rstdLocal, (float)0.0, calc_row_num);
    pipe_barrier(PIPE_V);
    for (uint32_t j = 0; j < j_max - 1; j++) {
      ComputeFormer(i_o, calc_row_num, j, rstdLocal, sumLocal, ubFactor);
    }
    // do tail
    ComputeFormer(i_o, calc_row_num, j_max - 1, rstdLocal, sumLocal, col_tail);
    ComputeRstd(rstdLocal, calc_row_num);

    for (uint32_t j = 0; j < j_max - 1; j++) {
      ComputeLatter(i_o, calc_row_num, j, rstdLocal, ubFactor);
    }
    ComputeLatter(i_o, calc_row_num, j_max - 1, rstdLocal, col_tail);
    outQueueRstd.EnQue<float>(rstdLocal);
    CopyOutRstd(i_o, calc_row_num);
  }

 private:
  __aicore__ inline void CopyInAndAdd(uint32_t i_idx, uint32_t j_idx, uint32_t num) {
    LocalTensor<T> x1x2_in = inQueueX.AllocTensor<T>();
    LocalTensor<T> x1_in = x1x2_in[0];
    LocalTensor<T> x2_in = x1x2_in[ubFactor];
    DataCopyCustom<T>(x1_in, x1Gm[i_idx * numCol + j_idx * ubFactor], num);
    DataCopyCustom<T>(x2_in, x2Gm[i_idx * numCol + j_idx * ubFactor], num);
    inQueueX.EnQue(x1x2_in);
    LocalTensor<T> x1x2Local = inQueueX.DeQue<T>();

    auto x1Local = x1x2Local[0];
    auto x2Local = x1x2Local[ubFactor];

    LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();

    if constexpr (is_same<T, half>::value) {
      LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();

      Add(xLocal, x1Local, x2Local, num);
      pipe_barrier(PIPE_V);
      Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, num);
      pipe_barrier(PIPE_V);
      // x1+x2 saved in x1_fp32
    } else if constexpr (is_same<T, bfloat16_t>::value) {
      LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
      LocalTensor<float> x2_fp32 = x1x2Local.template ReinterpretCast<float>();

      Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, num);
      pipe_barrier(PIPE_V);
      Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, num);
      pipe_barrier(PIPE_V);

      Add(x1_fp32, x1_fp32, x2_fp32, num);
      pipe_barrier(PIPE_V);
      Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, num);
      pipe_barrier(PIPE_V);
      // x1+x2 saved in x1_fp32
    } else {
      Add(x1Local, x1Local, x2Local, num);
      pipe_barrier(PIPE_V);
      Adds(xLocal, x1Local, (float)0.0, num);
      // x1+x2 saved in inQueueX
    }
    inQueueX.FreeTensor(x1x2Local);

    // copy out to workspace && x_out
    outQueueY.EnQue(xLocal);
    auto x_out = outQueueY.DeQue<T>();
    DataCopyCustom<T>(xGm[i_idx * numCol + j_idx * ubFactor], x_out, num);
    outQueueY.FreeTensor(x_out);
  }

  __aicore__ inline void ComputeFormer(uint32_t i_o_idx, uint32_t calc_row_num, uint32_t j_idx,
                                       LocalTensor<float>& rstdLocal, LocalTensor<float>& sumLocal, uint32_t num) {
    for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
      CopyInAndAdd(i_o_idx * rowFactor + i_i, j_idx, num);
      ComputeSum(i_i, sumLocal, num);
    }
    BlockReduceSumFP32(sumLocal, sumLocal, calc_row_num * NUM_PER_BLK_FP32);
    Add(rstdLocal, rstdLocal, sumLocal, calc_row_num);
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void ComputeSum(uint32_t i_i_idx, LocalTensor<float>& sumLocal, uint32_t num) {
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
      pipe_barrier(PIPE_V);
      Mul(sqx, x_fp32, x_fp32, num);
    } else {
      LocalTensor<T> xLocal = inQueueX.AllocTensor<float>();
      pipe_barrier(PIPE_V);
      Mul(sqx, xLocal, xLocal, num);
      inQueueX.FreeTensor(xLocal);
    }
    pipe_barrier(PIPE_V);
    Muls(sqx, sqx, avgFactor, num);
    pipe_barrier(PIPE_V);
    // 8 means 8 fp32 pre block
    ReduceSumFP32ToBlock(sumLocal[i_i_idx * 8], sqx, reduce_buf_local, num);
  }

  __aicore__ inline void ComputeRstd(LocalTensor<float> rstdLocal, uint32_t num) {
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
    Adds(rstdLocal, rstdLocal, epsilon, num);
    pipe_barrier(PIPE_V);
    Sqrt(rstdLocal, rstdLocal, num);
    Duplicate(reduce_buf_local, ONE, num);
    pipe_barrier(PIPE_V);
    Div(rstdLocal, reduce_buf_local, rstdLocal, num);
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void ComputeLatter(uint32_t i_o_idx, uint32_t calc_row_num, uint32_t j_idx,
                                       LocalTensor<float>& rstdLocal, uint32_t num) {
    CopyInGamma(j_idx, num);
    LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
    for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
      CopyInX(i_o_idx * rowFactor + i_i, j_idx, num);
      ComputeY(i_i, gammaLocal, rstdLocal, num);
      CopyOutY(i_o_idx * rowFactor + i_i, j_idx, num);
    }
    inQueueGamma.FreeTensor(gammaLocal);
  }

  __aicore__ inline void CopyInGamma(uint32_t j_idx, uint32_t num) {
    LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
    DataCopyCustom<T>(gammaLocal, gammaGm[j_idx * ubFactor], num);
    inQueueGamma.EnQue(gammaLocal);
  }

  __aicore__ inline void CopyInX(uint32_t i_idx, uint32_t j_idx, uint32_t num) {
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopyCustom<T>(xLocal, xGm[i_idx * numCol + j_idx * ubFactor], num);
    inQueueX.EnQue<T>(xLocal);
    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
      LocalTensor<T> xLocal = inQueueX.DeQue<T>();
      Cast(x_fp32, xLocal, RoundMode::CAST_NONE, num);
      pipe_barrier(PIPE_V);
      inQueueX.FreeTensor(xLocal);
    }
  }

  __aicore__ inline void ComputeY(uint32_t i_i_idx, LocalTensor<half>& gammaLocal, LocalTensor<float>& rstdLocal,
                                  uint32_t num) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = rstdLocal.GetValue(i_i_idx);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    pipe_barrier(PIPE_V);
    Muls(x_fp32, x_fp32, rstd_value, num);
    pipe_barrier(PIPE_V);
    LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
    Cast(yLocal, x_fp32, RoundMode::CAST_NONE, num);
    pipe_barrier(PIPE_V);
    Mul(yLocal, gammaLocal, yLocal, num);
    pipe_barrier(PIPE_V);
    outQueueY.EnQue<half>(yLocal);
  }

  __aicore__ inline void ComputeY(uint32_t i_i_idx, LocalTensor<float>& gammaLocal, LocalTensor<float>& rstdLocal,
                                  uint32_t num) {
    LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = rstdLocal.GetValue(i_i_idx);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    Muls(yLocal, xLocal, rstd_value, num);
    inQueueX.FreeTensor(xLocal);
    pipe_barrier(PIPE_V);
    Mul(yLocal, gammaLocal, yLocal, num);
    pipe_barrier(PIPE_V);
    outQueueY.EnQue<float>(yLocal);
  }

  __aicore__ inline void ComputeY(uint32_t i_i_idx, LocalTensor<bfloat16_t>& gammaLocal, LocalTensor<float>& rstdLocal,
                                  uint32_t num) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = rstdLocal.GetValue(i_i_idx);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    pipe_barrier(PIPE_V);
    Muls(x_fp32, x_fp32, rstd_value, num);
    pipe_barrier(PIPE_V);
    LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
    Cast(yLocal, x_fp32, RoundMode::CAST_RINT, num);
    pipe_barrier(PIPE_V);
    Cast(x_fp32, yLocal, RoundMode::CAST_NONE, num);
    pipe_barrier(PIPE_V);
    Cast(sqx, gammaLocal, RoundMode::CAST_NONE, num);
    pipe_barrier(PIPE_V);
    Mul(x_fp32, x_fp32, sqx, num);
    pipe_barrier(PIPE_V);
    Cast(yLocal, x_fp32, RoundMode::CAST_RINT, num);
    pipe_barrier(PIPE_V);
    outQueueY.EnQue<bfloat16_t>(yLocal);
  }

  __aicore__ inline void CopyOutY(uint32_t i_idx, uint32_t j_idx, uint32_t num) {
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopyCustom<T>(yGm[i_idx * numCol + j_idx * ubFactor], yLocal, num);
    outQueueY.FreeTensor(yLocal);
  }

  __aicore__ inline void CopyOutRstd(uint32_t i_o_idx, uint32_t num) {
    LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if __CCE_AICORE__ == 220
    DataCopyCustom<float>(rstdGm[i_o_idx * rowFactor], rstdLocal, num);
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
  TBuf<TPosition::VECCALC> sumBuf;
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

  int tempbufNum;
};
#endif  // _ADD_RMS_NORM_SPLIT_D_H_