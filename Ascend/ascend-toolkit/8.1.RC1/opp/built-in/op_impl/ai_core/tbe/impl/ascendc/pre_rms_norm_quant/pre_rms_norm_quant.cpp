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
 * \file pre_rms_norm_quant.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "../pre_rms_norm/kernel_common.h"

using AscendC::HardEvent;

template <bool FastComputeMode = false>
class PreRmsNormQuant {
 public:
  __aicore__ inline PreRmsNormQuant(__gm__ uint8_t* x, __gm__ uint8_t* res_in, __gm__ uint8_t* gamma, __gm__ uint8_t* beta,
                                     __gm__ uint8_t* y,
                                    __gm__ uint8_t* res, NormCommonTilingData& tiling_data)
      : num_core_(tiling_data.numCore),
        num_col_(tiling_data.numCol),
        avg_factor_(tiling_data.avgFactor),
        epsilon_(tiling_data.epsilon) {
    slice_size_ = tiling_data.sliceSize;
    num_slice_ = (num_col_ + slice_size_ - 1) / slice_size_;
    tail_size_ = num_col_ - (num_slice_ - 1) * slice_size_;
    uint32_t num_row = tiling_data.numRow;
    uint32_t row_work = (num_row + num_core_ - 1) / num_core_;
    if (block_idx != num_core_ - 1) {
      row_work_ = row_work;
    } else {
      row_work_ = num_row - (num_core_ - 1) * row_work;
    }
    gm_offset_ = static_cast<uint64_t>(row_work) * num_col_;
    quant_min_ = tiling_data.quantMin;
    float tiling_scale = (float)tiling_data.scale;
    assert(tiling_scale != 0, "Error: The scale parameter in tiling_data is set to zero!");
    input_scale_ = 1 / tiling_scale;
    input_offset_ = (float)tiling_data.offset;

    gm_x_.SetGlobalBuffer((__gm__ half*)x + AscendC::GetBlockIdx() * gm_offset_);
    gm_r_.SetGlobalBuffer((__gm__ half*)res_in + AscendC::GetBlockIdx() * gm_offset_);
    gm_g_.SetGlobalBuffer((__gm__ half*)gamma);
    gm_b_.SetGlobalBuffer((__gm__ half*)beta);
    gm_y_.SetGlobalBuffer((__gm__ int8_t*)y + AscendC::GetBlockIdx() * gm_offset_);
    gm_res_.SetGlobalBuffer((__gm__ half*)res + AscendC::GetBlockIdx() * gm_offset_);
    pipe.InitBuffer(fp16_x_que_, BUFFER_NUM, slice_size_ * sizeof(half));
    pipe.InitBuffer(fp16_res_in_que_, BUFFER_NUM, slice_size_ * sizeof(half));
    pipe.InitBuffer(fp16_gamma_que_, BUFFER_NUM, slice_size_ * sizeof(half));
    pipe.InitBuffer(fp16_beta_que_, BUFFER_NUM, slice_size_ * sizeof(half));
    pipe.InitBuffer(int8_y_que_, BUFFER_NUM, slice_size_ * sizeof(int8_t));
    pipe.InitBuffer(fp16_res_que_, BUFFER_NUM, slice_size_ * sizeof(half));
    pipe.InitBuffer(fp32_xy_buf_, slice_size_ * sizeof(float));
    pipe.InitBuffer(calc_buf_, slice_size_ * sizeof(float));

    // GetQuantInfo<float, float>(input_scale_, input_offset_, scale, offset, calc_buf_);
  }

  __aicore__ inline void Launch() {
    if constexpr (FastComputeMode) {
      FastCompute();
    } else {
      SliceCompute();
    }
  }

 private:
  __aicore__ inline void SliceCompute() {
    for (uint64_t pid = 0; pid < row_work_; pid++) {
      uint64_t row_offset = pid * num_col_;
      float squareSum = 0.0f;
      for (uint64_t sid = 0; sid < num_slice_; sid++) {
        uint64_t col_offset = row_offset + sid * slice_size_;
        uint32_t eleNum = (sid == (num_slice_ - 1)) ? tail_size_ : slice_size_;
        squareSum += ComputeSquareSum(col_offset, eleNum);
      }
      float rms = sqrt(avg_factor_ * squareSum + epsilon_);
      for (uint64_t sid = 0; sid < num_slice_; sid++) {
        uint64_t sliceOffset = sid * slice_size_;
        uint64_t totalOffset = row_offset + sliceOffset;
        uint32_t eleNum = (sid == (num_slice_ - 1)) ? tail_size_ : slice_size_;
        ComputeNorm(rms, totalOffset, sliceOffset, eleNum);
      }
    }
  }

  __aicore__ inline void FastCompute() {
    for (uint64_t pid = 0; pid < row_work_; pid++) {
      uint64_t offset = pid * num_col_;
      CopyInAll(offset, num_col_);
      Compute(offset);
      CopyOutAll(offset, num_col_);
    }
  }

 private:
  __aicore__ inline float ComputeSquareSum(uint64_t offset, uint32_t numel) {
    AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
    AscendC::LocalTensor<float> fp32_tmp = calc_buf_.Get<float>();

    CopyInAndCastF32(fp32_xy, gm_x_, fp16_x_que_, offset, numel);
    CopyInAndCastF32(fp32_tmp, gm_r_, fp16_res_in_que_, offset, numel);
    ComputeResidualAdd(fp32_xy, fp32_xy, fp32_tmp, numel);

    return ComputeSliceSquareSum(fp32_xy, fp32_tmp, fp32_tmp, numel);
  }

  __aicore__ inline void ComputeNorm(float rms, uint64_t totalOffset, uint64_t sliceOffset, uint32_t numel) {
    CopyIn(gm_g_, fp16_gamma_que_, sliceOffset, numel);
    CopyIn(gm_b_, fp16_beta_que_, sliceOffset, numel);
    AscendC::LocalTensor<half> g = fp16_gamma_que_.DeQue<half>();
    AscendC::LocalTensor<half> b = fp16_beta_que_.DeQue<half>();
    AscendC::LocalTensor<int8_t> int8_y = int8_y_que_.AllocTensor<int8_t>();

    AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
    AscendC::LocalTensor<float> fp32_tmp = calc_buf_.Get<float>();
    AscendC::LocalTensor<half> fp16_tmp = calc_buf_.Get<half>();

    CopyInAndCastF32(fp32_xy, gm_x_, fp16_x_que_, totalOffset, numel);
    CopyInAndCastF32(fp32_tmp, gm_r_, fp16_res_in_que_, totalOffset, numel);
    ComputeResidualAdd(fp32_xy, fp32_xy, fp32_tmp, numel);
    Cast16AndCopyOut(fp32_xy, gm_res_, fp16_res_que_, totalOffset, numel);
    ComputeRmsNorm(fp32_xy, fp32_xy, rms, g, b, fp32_tmp, numel);
    ComputeFp32ToI8Quant(int8_y, fp32_xy, fp16_tmp, (half)input_scale_, (half)input_offset_, quant_min_, numel);

    int8_y_que_.EnQue(int8_y);
    fp16_gamma_que_.FreeTensor(g);
    fp16_beta_que_.FreeTensor(b);
    CopyOut(gm_y_, int8_y_que_, totalOffset, numel);
  }

  __aicore__ inline void CopyInAll(uint64_t offset, uint32_t numel) {
    CopyIn(gm_x_, fp16_x_que_, offset, numel);
    CopyIn(gm_r_, fp16_res_in_que_, offset, numel);
    CopyIn(gm_g_, fp16_gamma_que_, 0, num_col_);
    CopyIn(gm_b_, fp16_beta_que_, 0, num_col_);
  }

  __aicore__ inline void Compute(uint64_t offset) {
    AscendC::LocalTensor<half> fp16_x = fp16_x_que_.DeQue<half>();
    AscendC::LocalTensor<half> fp16_res = fp16_res_in_que_.DeQue<half>();
    AscendC::LocalTensor<half> g = fp16_gamma_que_.DeQue<half>();
    AscendC::LocalTensor<half> b = fp16_beta_que_.DeQue<half>();
    AscendC::LocalTensor<int8_t> int8_y = int8_y_que_.AllocTensor<int8_t>();
    AscendC::LocalTensor<half> fp16_y = fp16_res_que_.AllocTensor<half>();

    AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
    AscendC::LocalTensor<float> fp32_tmp = calc_buf_.Get<float>();
    AscendC::LocalTensor<half> fp16_tmp = calc_buf_.Get<half>();

    CastFrom16To32(fp32_xy, fp16_x, num_col_);
    CastFrom16To32(fp32_tmp, fp16_res, num_col_);
    ComputeResidualAdd(fp32_xy, fp32_xy, fp32_tmp, num_col_);
    CastFrom32To16(fp16_y, fp32_xy, num_col_);

    float squareSum = ComputeSliceSquareSum(fp32_xy, fp32_tmp, fp32_tmp, num_col_);
    float rms = sqrt(squareSum * avg_factor_ + epsilon_);
    ComputeRmsNorm(fp32_xy, fp32_xy, rms, g, b, fp32_tmp, num_col_);
    ComputeFp32ToI8Quant(int8_y, fp32_xy, fp16_tmp, (half)input_scale_, (half)input_offset_, quant_min_, num_col_);

    int8_y_que_.EnQue(int8_y);
    fp16_res_que_.EnQue(fp16_y);
    fp16_x_que_.FreeTensor(fp16_x);
    fp16_res_in_que_.FreeTensor(fp16_res);
    fp16_gamma_que_.FreeTensor(g);
    fp16_beta_que_.FreeTensor(b);
  }

  __aicore__ inline void CopyOutAll(uint64_t offset, uint32_t numel) {
    CopyOut(gm_y_, int8_y_que_, offset, numel);
    CopyOut(gm_res_, fp16_res_que_, offset, numel);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16_x_que_;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16_res_in_que_;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16_gamma_que_;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16_beta_que_;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> int8_y_que_;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> fp16_res_que_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_xy_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_;
  AscendC::GlobalTensor<half> gm_x_;
  AscendC::GlobalTensor<half> gm_r_;
  AscendC::GlobalTensor<half> gm_g_;
  AscendC::GlobalTensor<half> gm_b_;
  AscendC::GlobalTensor<int8_t> gm_y_;
  AscendC::GlobalTensor<half> gm_res_;
  uint32_t num_core_{0};    // 一共激活多少AICORE
  uint32_t num_col_{0};     // 输入的列数
  uint32_t row_work_{0};    // 需要计算多少行
  uint32_t row_step_{0};    // 除最后一次，每次搬入多少行
  uint32_t row_tail_{0};    // 最后一次搬入多少行数据
  uint64_t gm_offset_{0};   // GM数据起始位置偏移量
  float avg_factor_{1.0};   // num_col_的倒数
  float input_scale_{1.0};  // 非对称量化系数
  float input_offset_{0};   // 非对称量化偏移适配高精度
  float epsilon_{1e-12f};   // norm平滑参数
  float quant_min_{-128.0};
  uint32_t slice_size_{0};  // 每一行切分的大小
  int32_t num_slice_{0};
  int32_t tail_size_{0};
};

extern "C" __global__ __aicore__ void pre_rms_norm_quant(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, 
                                                         GM_ADDR res_in, GM_ADDR y, GM_ADDR res_out, GM_ADDR workspace, GM_ADDR tiling){
  GET_TILING_DATA(tiling_data, tiling);
  if (TILING_KEY_IS(0b0000000001)) { // 0b0_0_0_0_000001
    PreRmsNormQuant<true> kernel(x, res_in, gamma, beta, y, res_out, tiling_data);
    kernel.Launch();
  }
  if (TILING_KEY_IS(0b0001000001)) {   // 0b0_0_0_1_000001
    PreRmsNormQuant<false> kernel(x, res_in, gamma, beta, y, res_out, tiling_data);
    kernel.Launch();
  }
}