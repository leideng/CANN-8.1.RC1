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
 * \file add_rms_norm_multi_n.h
 * \brief
 */
#ifndef _ADD_RMS_NORM_MULTI_N_H_
#define _ADD_RMS_NORM_MULTI_N_H_
#include "../rms_norm/rms_norm_base.h"

using namespace AscendC;

template <typename T>
class KernelAddRmsNormMultiN {
 public:
  __aicore__ inline KernelAddRmsNormMultiN(TPipe *pipe) {Ppipe = pipe;
  }
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x,
                              const AddRMSNormTilingData* tiling) {
    ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
    this->numRow = tiling->num_row;
    this->numCol = tiling->num_col;
    uint32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
    this->numColAlign = AlignUp(numCol, numPerBlock);
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

    // pipe alloc memory to queue, the unit is Bytes
    Ppipe->InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
    Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, numColAlign * sizeof(T));
    Ppipe->InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
#if __CCE_AICORE__ == 220
    Ppipe->InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * NUM_PER_BLK_FP32 * sizeof(float));
#else
    Ppipe->InitBuffer(rstdBuf, rowFactor * NUM_PER_BLK_FP32 * sizeof(float));
#endif
    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
    }
    Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
    Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
    Ppipe->InitBuffer(offsetBuf, rowFactor * NUM_PER_BLK_FP32 * sizeof(uint32_t));
  }

  __aicore__ inline void Process() {
    CopyInGamma();
    LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
    LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
    for (uint32_t i = 0; i < rowFactor; i++) {
      Duplicate(offsetLocal[i * NUM_PER_BLK_FP32], i * ONE_BLK_SIZE, NUM_PER_BLK_FP32);
    }
    uint32_t i_o_max = CeilDiv(rowWork, rowFactor);
    uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;

    for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
      SubProcess(i_o, rowFactor, gammaLocal);
    }
    SubProcess(i_o_max - 1, row_tail, gammaLocal);
    inQueueGamma.FreeTensor(gammaLocal);
  }

  __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, LocalTensor<T>& gammaLocal) {
    uint32_t gm_bias = i_o * rowFactor * numCol;
    CopyInX(gm_bias, calc_row_num);
    LocalTensor<T> xLocal = ComputeX(calc_row_num);
    CopyOutX(gm_bias, calc_row_num);
#if __CCE_AICORE__ == 220
    LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
    ComputeRstd(xLocal, rstdLocal, calc_row_num);
    outQueueRstd.EnQue<float>(rstdLocal);
    CopyOutRstd(i_o * rowFactor, calc_row_num);
#else
    LocalTensor<float> rstdLocal = rstdBuf.Get<float>();
    ComputeRstd(xLocal, rstdLocal, calc_row_num);
#endif
    ComputeY(xLocal, gammaLocal, rstdLocal, calc_row_num);
    CopyOutY(gm_bias, calc_row_num);
  }

 private:
  __aicore__ inline void CopyInX(uint32_t gm_bias, uint32_t calc_row_num) {
    LocalTensor<T> x1Local = inQueueX.AllocTensor<T>();
    DataCopyCustom<T>(x1Local, x1Gm[gm_bias], calc_row_num * numCol);
    inQueueX.EnQue(x1Local);
    LocalTensor<T> x2Local = inQueueX.AllocTensor<T>();
    DataCopyCustom<T>(x2Local, x2Gm[gm_bias], calc_row_num * numCol);
    inQueueX.EnQue(x2Local);
  }

  __aicore__ inline LocalTensor<T> ComputeX(uint32_t calc_row_num) {
    uint32_t calc_num = calc_row_num * numColAlign;
    LocalTensor<T> x1Local = inQueueX.DeQue<T>();
    LocalTensor<T> x2Local = inQueueX.DeQue<T>();
    LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();
    Add(xLocal, x1Local, x2Local, calc_num);
    inQueueX.FreeTensor(x1Local);
    inQueueX.FreeTensor(x2Local);
    outQueueY.EnQue(xLocal);
    pipe_barrier(PIPE_V);
    return xLocal;
  }

  __aicore__ inline void CopyOutX(uint32_t gm_bias, uint32_t calc_row_num) {
    // CopyOut x1 + x2
    auto x_out = outQueueY.DeQue<T>();
    DataCopyCustom<T>(xGm[gm_bias], x_out, calc_row_num * numCol);
    outQueueY.FreeTensor(x_out);
  }

  __aicore__ inline void CopyInGamma() {
    LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
    DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
    inQueueGamma.EnQue(gammaLocal);
  }

  __aicore__ inline void ComputeRstd(LocalTensor<T> xLocal, LocalTensor<float> rstdLocal, uint32_t calc_row_num) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
    Cast(x_fp32, xLocal, RoundMode::CAST_NONE, calc_row_num * numColAlign);
    pipe_barrier(PIPE_V);

    Mul(sqx, x_fp32, x_fp32, calc_row_num * numColAlign);
    pipe_barrier(PIPE_V);

    Muls(sqx, sqx, avgFactor, calc_row_num * numColAlign);
    pipe_barrier(PIPE_V);

    for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
      ReduceSumCustom(rstdLocal[i_i * NUM_PER_BLK_FP32], sqx[i_i * numColAlign], reduce_buf_local, numCol);
    }
    Adds(rstdLocal, rstdLocal, epsilon, calc_row_num * NUM_PER_BLK_FP32);
    pipe_barrier(PIPE_V);

    Sqrt(rstdLocal, rstdLocal, calc_row_num * NUM_PER_BLK_FP32);
    Duplicate(reduce_buf_local, ONE, NUM_PER_BLK_FP32);
    pipe_barrier(PIPE_V);

    int32_t repeatTimes = calc_row_num * NUM_PER_BLK_FP32 / NUM_PER_REP_FP32;
    int32_t tailCount = calc_row_num * NUM_PER_BLK_FP32 % NUM_PER_REP_FP32;
    int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;

    if (likely(repeatTimes > 0)) {
      Div(rstdLocal, reduce_buf_local, rstdLocal, NUM_PER_REP_FP32, repeatTimes,
          {1, 0, 1, DEFAULT_REPEAT_STRIDE, 0, DEFAULT_REPEAT_STRIDE});
    }
    if (unlikely(tailCount != 0)) {
      Div(rstdLocal[bodyCount], reduce_buf_local, rstdLocal[bodyCount], tailCount, 1,
          {1, 0, 1, DEFAULT_REPEAT_STRIDE, 0, DEFAULT_REPEAT_STRIDE});
    }
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void ComputeY(LocalTensor<T> xLocal, LocalTensor<T> gammaLocal, LocalTensor<float> rstdLocal,
                                  uint32_t calc_row_num) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
    Gather(rstdLocal, rstdLocal, offsetLocal, ZERO_UINT, calc_row_num * NUM_PER_BLK_FP32);
    pipe_barrier(PIPE_V);
    int32_t repeatTimes = numCol / NUM_PER_REP_FP32;
    int32_t tailCount = numCol % NUM_PER_REP_FP32;
    int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
    for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
      if (likely(repeatTimes > 0)) {
        Mul(x_fp32[i_i * numColAlign], x_fp32[i_i * numColAlign], rstdLocal[i_i * NUM_PER_BLK_FP32], NUM_PER_REP_FP32,
            repeatTimes, {1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0});
      }
      if (unlikely(tailCount != 0)) {
        Mul(x_fp32[i_i * numColAlign + bodyCount], x_fp32[i_i * numColAlign + bodyCount],
            rstdLocal[i_i * NUM_PER_BLK_FP32], tailCount, 1,
            {1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0});
      }
    }
    pipe_barrier(PIPE_V);
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    Cast(yLocal, x_fp32, RoundMode::CAST_NONE, calc_row_num * numColAlign);
    pipe_barrier(PIPE_V);

    for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
      Mul(yLocal[i_i * numColAlign], gammaLocal, yLocal[i_i * numColAlign], numCol);
    }
    pipe_barrier(PIPE_V);
    outQueueY.EnQue<T>(yLocal);
  }

  __aicore__ inline void CopyOutY(uint32_t progress, uint32_t calc_row_num) {
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopyCustom<T>(yGm[progress], yLocal, calc_row_num * numCol);
    outQueueY.FreeTensor(yLocal);
  }

#if __CCE_AICORE__ == 220
  __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num) {
    LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
    DataCopyParams copyParams;
    copyParams.blockLen = sizeof(float);
    copyParams.blockCount = num;
    DataCopyPad(rstdGm[outer_progress], rstdLocal, copyParams);
    outQueueRstd.FreeTensor(rstdLocal);
  }
#endif

 private:
  TPipe *Ppipe = nullptr;
  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
  TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX;
  // create queues for output, in this case depth is equal to buffer num
#if __CCE_AICORE__ == 220
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;
#else
  TBuf<TPosition::VECCALC> rstdBuf;
#endif
  TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;

  TBuf<TPosition::VECCALC> xFp32Buf;
  TBuf<TPosition::VECCALC> sqxBuf;
  TBuf<TPosition::VECCALC> reduceFp32Buf;
  TBuf<TPosition::VECCALC> offsetBuf;
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
  uint32_t numColAlign;
  int32_t blockIdx_;
  uint32_t rowWork = 1;
};
#endif  // ADD_RMS_NORM_H_