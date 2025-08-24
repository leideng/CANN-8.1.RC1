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
 * \file pre_rms_norm.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_common.h"

using AscendC::HardEvent;

namespace{
  constexpr uint32_t NUM_TWO = 2;
  constexpr uint32_t NUM_THREE = 3;
  constexpr uint32_t NUM_FOUR = 4;
  constexpr uint32_t BLOCK_BYTE = 32;
  constexpr uint32_t FP32_PER_REPEAT = 64;
  constexpr uint32_t FP16_PER_REPEAT = 128;
}

template <typename T, bool EN_BETA>
class PreRmsNormShort {
 public:
  __aicore__ inline PreRmsNormShort(__gm__ uint8_t* x, __gm__ uint8_t* bias, __gm__ uint8_t* res_in, 
                                    __gm__ uint8_t* g, __gm__ uint8_t* y, __gm__ uint8_t* res_out,
                                    NormCommonTilingData& tiling_data) {
    uint32_t numRow = tiling_data.numRow;
    numCore_ = tiling_data.numCore;
    numCol_ = tiling_data.numCol;
    avgFactor_ = tiling_data.avgFactor;
    epsilon_ = tiling_data.epsilon;
    sliceSize_ = tiling_data.sliceSize;
    precisionMode_ = tiling_data.highPrecisionMode;
    uint32_t rowWork = (numRow + numCore_ - 1) / numCore_;

    if (AscendC::GetBlockIdx() != numCore_ - 1) {
      rowWork_ = rowWork;
    } else {
      rowWork_ = numRow - (numCore_ - 1) * rowWork;
    }
    gm_offset_ = static_cast<uint64_t>(rowWork) * numCol_;

    numColAlignFp32 = (numCol_ + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
    numColAlignFp16 = (numCol_ + FP16_PER_REPEAT - 1) / FP16_PER_REPEAT * FP16_PER_REPEAT;

    gm_x_.SetGlobalBuffer((__gm__ T*)x + AscendC::GetBlockIdx() * gm_offset_);
    if constexpr (EN_BETA) {
      gm_bias_.SetGlobalBuffer((__gm__ T*)bias);
      pipe.InitBuffer(calc_buf_, NUM_TWO * sliceSize_ * sizeof(float));
      pipe.InitBuffer(fp16_x_buf_, NUM_FOUR * sliceSize_ * sizeof(T));  // x,res_in,gamma,bias
    } else {
      pipe.InitBuffer(calc_buf_, 1 * sliceSize_ * sizeof(float));
      pipe.InitBuffer(fp16_x_buf_, NUM_THREE * sliceSize_ * sizeof(T));
    }
    gm_g_.SetGlobalBuffer((__gm__ T*)g);
    gm_res_in_.SetGlobalBuffer((__gm__ T*)res_in + AscendC::GetBlockIdx() * gm_offset_);
    gm_y_.SetGlobalBuffer((__gm__ T*)y + AscendC::GetBlockIdx() * gm_offset_);
    gm_res_out_.SetGlobalBuffer((__gm__ T*)res_out + AscendC::GetBlockIdx() * gm_offset_);

    pipe.InitBuffer(fp32_xy_buf_, sliceSize_ * sizeof(float));
    pipe.InitBuffer(fp16_out_, sliceSize_ * sizeof(T));

    AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
    DataCopyCustom<T>(fp16_x, gm_x_[0], numColAlignFp16);
    DataCopyCustom<T>(fp16_x[sliceSize_], gm_res_in_[0], numColAlignFp16);
    DataCopy(fp16_x[sliceSize_ * NUM_TWO], gm_g_, numColAlignFp16);
  }

  __aicore__ inline void Launch() {
    AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
    AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
    AscendC::LocalTensor<T> fp16_res_in_ = fp16_x[sliceSize_];

    if constexpr (EN_BETA) {
      AscendC::LocalTensor<T> fp16_bias = fp16_x[sliceSize_ * NUM_THREE];
      AscendC::LocalTensor<float> fp32_bias = buf[sliceSize_];
      BiasIn(fp16_x, fp16_bias, fp32_bias, gm_bias_, numColAlignFp16);
    }

    uint64_t pid = 0;
    while (pid < rowWork_) {
      uint64_t offset = pid * numCol_;
      if (pid != 0) {
        CopyInXResIn(fp16_x, fp16_res_in_, gm_x_, gm_res_in_, offset, numColAlignFp16);
      }
      AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

      Compute(offset);

      AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      CopyOut(offset, numCol_);
      AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
      AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
      ++pid;
    }
  }

 private:
  __aicore__ inline void Compute(uint32_t offset) {
    AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
    AscendC::LocalTensor<T> fp16_res_in = fp16_x[sliceSize_];
    AscendC::LocalTensor<T> fp16_gamma = fp16_x[sliceSize_ * NUM_TWO];
    AscendC::LocalTensor<float> fp32_reduce_workspace = fp16_out_.Get<float>();
    AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
    AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
    AscendC::LocalTensor<T> out_buf = fp16_out_.Get<T>();
    AscendC::LocalTensor<float> sqx = buf[0];

    if constexpr (EN_BETA) {
      AscendC::LocalTensor<T> fp16_bias = fp16_x[sliceSize_ * NUM_THREE];
      AscendC::LocalTensor<float> buf_bias = buf[sliceSize_];
      AddResBiasAndCast<T>(fp16_x, fp16_res_in, fp16_bias, fp32_xy, buf, buf_bias, numCol_);
    } else {
      AddResAndCast<T>(fp16_x, fp16_res_in, fp32_xy, buf, numCol_);
    }
    CastFrom32To16(fp16_x, fp32_xy, numCol_);
    AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

    DataCopyCustom<T>(gm_res_out_[offset], fp16_x, numCol_);
    AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    FigureOutNorm(sqx, fp32_xy, fp32_reduce_workspace, avgFactor_, epsilon_, numCol_, numColAlignFp32);
    MultiplyGamma(fp16_gamma, sqx, fp32_xy, out_buf, numCol_, numColAlignFp32, numColAlignFp16, precisionMode_);
  }

  __aicore__ inline void CopyOut(uint32_t offset, uint32_t numel) {
    AscendC::LocalTensor<T> out_buf = fp16_out_.Get<T>();
    DataCopyCustom<T>(gm_y_[offset], out_buf, numel);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_x_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_y_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_res_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_xy_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_, fp16_out_;
  AscendC::GlobalTensor<T> gm_x_;
  AscendC::GlobalTensor<T> gm_bias_;
  AscendC::GlobalTensor<T> gm_res_in_;
  AscendC::GlobalTensor<T> gm_g_;
  AscendC::GlobalTensor<T> gm_y_;
  AscendC::GlobalTensor<T> gm_res_out_;
  NormCommonTilingData tiling_data_;
  uint32_t numCore_{0};    // 一共激活多少AICORE
  uint32_t numCol_{0};     // 输入的列数
  uint32_t rowWork_{0};    // 需要计算多少行
  uint32_t rowStep_{0};    // 除最后一次，每次搬入多少行 rowTail_
  uint32_t rowTail_{0};    // 最后一次搬入多少行数据
  uint64_t gm_offset_{0};  // GM数据起始位置偏移量
  uint32_t sliceSize_{0};  // 每一行切分的大小
  float avgFactor_{1.0f};  // num_col_的倒数
  float epsilon_{1e-12f};  // norm平滑参数
  uint32_t numColAlignFp32{64};
  uint32_t numColAlignFp16{128};
  uint32_t precisionMode_{0};
};

template <typename T, bool EN_BETA>
class PreRmsNormLong {
 public:
  __aicore__ inline PreRmsNormLong(__gm__ uint8_t* x, __gm__ uint8_t* bias, __gm__ uint8_t* res_in,
                                   __gm__ uint8_t* g, __gm__ uint8_t* y, __gm__ uint8_t* res_out,
                                   NormCommonTilingData& tiling_data) {
    uint32_t numRow = tiling_data.numRow;
    numCore_ = tiling_data.numCore;
    numCol_ = tiling_data.numCol;
    precisionMode_ = tiling_data.highPrecisionMode;
    avgFactor_ = tiling_data.avgFactor;
    epsilon_ = tiling_data.epsilon;
    // avgFactor_ = *reinterpret_cast<float*>(&tiling_data.avgFactor);
    // epsilon_ = *reinterpret_cast<float*>(&tiling_data.epsilon);
    sliceSize_ = tiling_data.sliceSize;
    uint32_t rowWork = (numRow + numCore_ - 1) / numCore_;
    if (AscendC::GetBlockIdx() != numCore_ - 1) {
      rowWork_ = rowWork;
    } else {
      rowWork_ = numRow - (numCore_ - 1) * rowWork;
    }
#if __CCE_AICORE__ != 220
    if ((numCol_ % sliceSize_) * sizeof(T) < BLOCK_BYTE && (numCol_ % sliceSize_) != 0) {
      sliceSizeTmp_ = sliceSize_ - ((BLOCK_BYTE / sizeof(T)) - (numCol_ % sliceSize_));
    } else {
      sliceSizeTmp_ = sliceSize_;
    }
#else
    sliceSizeTmp_ = sliceSize_;
#endif
    numSlice_ = (numCol_ + sliceSizeTmp_ - 1) / sliceSizeTmp_;
    tailSize_ = numCol_ - (numSlice_ - 1) * sliceSizeTmp_;
    gm_offset_ = static_cast<uint64_t>(rowWork) * numCol_;
    gm_x_.SetGlobalBuffer((__gm__ T*)x + AscendC::GetBlockIdx() * gm_offset_);
    if constexpr (EN_BETA) {
      gm_bias_.SetGlobalBuffer((__gm__ T*)bias);
      pipe.InitBuffer(calc_buf_, NUM_TWO * sliceSize_ * sizeof(float));
      pipe.InitBuffer(fp16_x_buf_, NUM_FOUR * sliceSize_ * sizeof(T));

    } else {
      pipe.InitBuffer(calc_buf_, 1 * sliceSize_ * sizeof(float));
      pipe.InitBuffer(fp16_x_buf_, NUM_THREE * sliceSize_ * sizeof(T));
    }
    gm_res_in_.SetGlobalBuffer((__gm__ T*)res_in + AscendC::GetBlockIdx() * gm_offset_);
    gm_g_.SetGlobalBuffer((__gm__ T*)g);
    gm_y_.SetGlobalBuffer((__gm__ T*)y + AscendC::GetBlockIdx() * gm_offset_);
    gm_res_out_.SetGlobalBuffer((__gm__ T*)res_out + AscendC::GetBlockIdx() * gm_offset_);

    pipe.InitBuffer(fp32_xy_buf_, sliceSize_ * sizeof(float));
    pipe.InitBuffer(fp16_out_, sliceSize_ * sizeof(T));
  }

  __aicore__ inline void Launch() {
    AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
    AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
    AscendC::LocalTensor<T> fp16_res_in_ = fp16_x[sliceSize_];
    AscendC::LocalTensor<T> fp16_gamma = fp16_x[NUM_TWO * sliceSize_];
    uint64_t pid = 0;
    while (pid < rowWork_) {
      uint64_t rowOffset = pid * numCol_;
      uint32_t numEle = sliceSizeTmp_;
      squareSum_ = 0.0f;
      for (uint64_t sid = 0; sid < numSlice_; sid++) {
        uint64_t colOffset = rowOffset + sid * sliceSizeTmp_;
        if ((sid == (numSlice_ - 1)) && (tailSize_ != 0)) {
          numEle = tailSize_;
        }
        numelAlignFp32 = (numEle + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
        numelAlignFp16 = (numEle + FP16_PER_REPEAT - 1) / FP16_PER_REPEAT * FP16_PER_REPEAT;

        CopyInXResIn(fp16_x, fp16_res_in_, gm_x_, gm_res_in_, colOffset, numelAlignFp16);
        if constexpr (EN_BETA) {
          CopyInBias(sid * sliceSizeTmp_);
        }
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        squareSum_ += ComputeSquareSum(numEle, sid, colOffset);
      }
      numEle = sliceSizeTmp_;
      float factor = avgFactor_ * squareSum_ + epsilon_;
      for (uint64_t sid = 0; sid < numSlice_; sid++) {
        uint64_t colOffset = rowOffset + sid * sliceSizeTmp_;
        if ((sid == (numSlice_ - 1)) && (tailSize_ != 0)) {
          numEle = tailSize_;
        }
        numelAlignFp32 = (numEle + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
        numelAlignFp16 = (numEle + FP16_PER_REPEAT - 1) / FP16_PER_REPEAT * FP16_PER_REPEAT;

        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        CopyInXResIn(fp16_x, fp16_res_in_, gm_x_, gm_res_in_, colOffset, numelAlignFp16);
        CopyInG(fp16_gamma, gm_g_, sid * sliceSizeTmp_, numelAlignFp16);
        if constexpr (EN_BETA) {
          CopyInBias(sid * sliceSizeTmp_);
        }
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        ComputeNorm(factor, numEle);
        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        CopyOut(colOffset, numEle);
        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
      }
      pid++;
    }
  }

 private:
  __aicore__ inline void CopyInBias(uint64_t offset) {
    AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
    AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
    AscendC::LocalTensor<float> fp32_bias = buf[sliceSize_];
    DataCopy(fp16_x[sliceSize_ * NUM_THREE], gm_bias_[offset], numelAlignFp16);
    AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Cast(fp32_bias, fp16_x[sliceSize_ * NUM_THREE], AscendC::RoundMode::CAST_NONE, numelAlignFp16);
  }
  __aicore__ inline float ComputeSquareSum(uint32_t numel, uint32_t sid, uint64_t colOffset) {
    AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
    AscendC::LocalTensor<T> fp16_res_in = fp16_x[sliceSize_];
    AscendC::LocalTensor<float> fp32_reduce_workspace = fp16_out_.Get<float>();
    AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
    AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
    AscendC::LocalTensor<float> sqx = buf[0];
    AscendC::LocalTensor<float> bias = buf[sliceSize_];

    if constexpr (EN_BETA) {
      AscendC::LocalTensor<T> fp16_bias = fp16_x[sliceSize_ * NUM_THREE];
      AscendC::LocalTensor<float> buf_bias = buf[sliceSize_];
      AddResBiasAndCast<T>(fp16_x, fp16_res_in, fp16_bias, fp32_xy, buf, buf_bias, numel);
    } else {
      AddResAndCast<T>(fp16_x, fp16_res_in, fp32_xy, buf, numel);
    }
    // fp32_xy = x + res_in
    CastFrom32To16(fp16_x, fp32_xy, numel);
    AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    DataCopyCustom<T>(gm_res_out_[colOffset], fp16_x, numel);
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    Mul(sqx, fp32_xy, fp32_xy, numelAlignFp32);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    ReduceSumCustom(sqx, sqx, fp32_reduce_workspace, numel);
    AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
    return sqx.GetValue(0);
  }

  __aicore__ void ComputeNorm(float sqs, uint32_t numel) {
    AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
    AscendC::LocalTensor<T> fp16_res_in = fp16_x[sliceSize_];
    AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
    AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
    AscendC::LocalTensor<T> out_buf = fp16_out_.Get<T>();
    AscendC::LocalTensor<float> sqx = buf[0];
    AscendC::LocalTensor<T> fp16_gamma = fp16_x[NUM_TWO * sliceSize_];

    if constexpr (EN_BETA) {
      AscendC::LocalTensor<T> fp16_bias = fp16_x[sliceSize_ * NUM_THREE];
      AscendC::LocalTensor<float> buf_bias = buf[sliceSize_];
      AddResBiasAndCast<T>(fp16_x, fp16_res_in, fp16_bias, fp32_xy, buf, buf_bias, numel);
    } else {
      AddResAndCast<T>(fp16_x, fp16_res_in, fp32_xy, buf, numel);
    }
    
    float factor = (sqs != 0) ? 1 / sqs : 1.0f;  // 避免除0错误
    AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);

    AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(sqx, factor, AscendC::DEFAULT_REPEAT_STRIDE);
    AscendC::PipeBarrier<PIPE_V>();

    Sqrt(sqx, sqx, AscendC::DEFAULT_REPEAT_STRIDE);
    AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);

    AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
    factor = sqx.GetValue(0);
    Muls(fp32_xy, fp32_xy, factor, numelAlignFp32);
    AscendC::PipeBarrier<PIPE_V>();
    MultiplyGamma(fp16_gamma, sqx, fp32_xy, out_buf, numel, numelAlignFp32, numelAlignFp16, precisionMode_);
  }
  __aicore__ inline void CopyOut(uint32_t offset, uint32_t numel) {
    AscendC::LocalTensor<T> out_buf = fp16_out_.Get<T>();
    DataCopyCustom<T>(gm_y_[offset], out_buf, numel);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_x_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_res_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_y_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_xy_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_, fp16_out_;
  AscendC::GlobalTensor<T> gm_x_;
  AscendC::GlobalTensor<T> gm_bias_;
  AscendC::GlobalTensor<T> gm_g_;
  AscendC::GlobalTensor<T> gm_res_in_;
  AscendC::GlobalTensor<T> gm_y_;
  AscendC::GlobalTensor<T> gm_res_out_;
  uint32_t numCore_{0};       // 一共激活多少AICORE
  uint32_t numCol_{0};        // 输入的列数
  uint32_t rowStep_{0};       // 除最后一次，每次搬入多少行
  uint32_t rowWork_{0};       // 需要计算多少行
  uint32_t rowTail_{0};       // 最后一次搬入多少行数据
  uint64_t gm_offset_{0};     // GM数据起始位置偏移量
  uint32_t sliceSize_{0};     // 每一行切分的大小
  uint32_t sliceSizeTmp_{0};  // 每一行切分的大小
  float epsilon_{1e-12f};     // norm平滑参数
  uint32_t numSlice_{0};
  uint32_t tailSize_{0};
  float avgFactor_{1.0f};  // num_col_的倒数
  float squareSum_{0.0f};
  uint32_t numelAlignFp32{64};
  uint32_t numelAlignFp16{32};
  uint32_t precisionMode_{0};
};

extern "C" __global__ __aicore__ void pre_rms_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR bias, GM_ADDR res_in, GM_ADDR y,
                                                   GM_ADDR res_out, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  if (TILING_KEY_IS(0b0100000001)) {  //0b0_1_0_0_000001
    PreRmsNormShort<half, true> kernel(x, bias, res_in, gamma, y, res_out, tiling_data);
    kernel.Launch();
  }
  if (TILING_KEY_IS(0b0000000001)) {  // 0b0_0_0_0_000001
    PreRmsNormShort<half, false> kernel(x, bias, res_in, gamma, y, res_out, tiling_data);
    kernel.Launch();
  }
  if (TILING_KEY_IS(0b0101000001)) {  // 0b0_1_0_1_000001
    PreRmsNormLong<half, true> kernel(x, bias, res_in, gamma, y, res_out, tiling_data);
    kernel.Launch();
  }
  if (TILING_KEY_IS(0b0001000001)) {  // 0b0_0_0_1_000001
    PreRmsNormLong<half, false> kernel(x, bias, res_in, gamma, y, res_out, tiling_data);
    kernel.Launch();
  }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
  if (TILING_KEY_IS(0b0100011011)) {    // 0b0_1_0_0_011011
    PreRmsNormShort<bfloat16_t, true> kernel(x, bias, res_in, gamma, y, res_out, tiling_data);
    kernel.Launch();
  }
  if (TILING_KEY_IS(0b0000011011)) {  // 0b0_0_0_0_011011
    PreRmsNormShort<bfloat16_t, false> kernel(x, bias, res_in, gamma, y, res_out, tiling_data);
    kernel.Launch();
  }
  if (TILING_KEY_IS(0b0101011011)) { // 0b0_1_0_1_011011
    PreRmsNormLong<bfloat16_t, true> kernel(x, bias, res_in, gamma, y, res_out, tiling_data);
    kernel.Launch();
  }
  if (TILING_KEY_IS(0b0001011011)) { // 0b0_0_0_1_011011
    PreRmsNormLong<bfloat16_t, false> kernel(x, bias, res_in, gamma, y, res_out, tiling_data);
    kernel.Launch();
  }
#endif
}