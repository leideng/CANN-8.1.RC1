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
 * \file trans_quant_param_v2.h
 * \brief
 */
#ifndef TRANS_QUANT_PARAM_V2_H
#define TRANS_QUANT_PARAM_V2_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "../inc/safe_data_copy.h"

namespace AscendC {

constexpr uint64_t DEQ_SCALE_MUL = 0xFFFFE000;
constexpr uint64_t QUANT_SCALE = 0x1ULL << 46;
constexpr uint64_t QUANT_MASK_0 = 0x1FFULL;
constexpr int32_t UB_ALIGN_SIZE = 32;
constexpr int32_t OFFSET_DEVIATION = 37;
constexpr int32_t SPLIT_SIZE = 65536;
constexpr int32_t MAX_INT9 = 255;
constexpr int32_t MIN_INT9 = -256;

class TransQuantParamV2 {
 public:
  __aicore__ inline TransQuantParamV2(){};
  __aicore__ inline void Init(GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace,
                              const TransQuantParamV2TilingData* tilingData, TPipe* tPipe) {
    if ASCEND_IS_AIC {
      return;
    }
    scaleLength_ = tilingData->scaleLength;
    offsetLength_ = tilingData->offsetLength;
    // init global buffer
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale);
    offsetGm_.SetGlobalBuffer((__gm__ float*)offset);
    yGm_.SetGlobalBuffer((__gm__ uint64_t*)y);
    pipe_ = tPipe;
  }

  /** main logical function
   */
  __aicore__ inline void Process() {
    if ASCEND_IS_AIC {
      return;
    }
    if (GetBlockIdx() > 0) {return;}
    CalOffsetValueShapeOne();
    CalScaleValueShapeOne();
    uint32_t eachLength = SPLIT_SIZE / sizeof(uint64_t);
    uint32_t loops = Max<uint32_t>(scaleLength_, offsetLength_) / eachLength;
    uint32_t tailLength = Max<uint32_t>(scaleLength_, offsetLength_) - loops * eachLength;
    uint32_t alignedLength = Align<uint32_t>(eachLength * sizeof(float), UB_ALIGN_SIZE) / sizeof(float);
    pipe_->InitBuffer(offsetUb_, 2 * alignedLength * sizeof(float)); //need 2 times space for float and int32_t tensor
    pipe_->InitBuffer(resUb_, SPLIT_SIZE);
    LocalTensor<uint64_t> resTensor = resUb_.Get<uint64_t>(eachLength);
    DataCopyParams ub2GmParams {1, SPLIT_SIZE / UB_ALIGN_SIZE, 0, 0};
    for (uint32_t loopidx = 0; loopidx < loops; ++loopidx) {
      for (uint32_t idx = 0; idx < eachLength; ++idx) {
        CalQuantPreScale(eachLength * loopidx + idx, idx, resTensor);
        if (offsetLength_ == 1) {
          resTensor.SetValue(idx, resTensor.GetValue(idx) | offetInt9Bit_);
        }
      }
      if (offsetLength_ > 1) {
        CalOffset(eachLength, eachLength * loopidx, resTensor);
      }
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      DataCopy(yGm_[eachLength * loopidx], resTensor, ub2GmParams);
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
    for (uint32_t idx = 0; idx < tailLength; ++idx) {
      CalQuantPreScale(eachLength * loops + idx, idx, resTensor);
      if (offsetLength_ == 1) {
        resTensor.SetValue(idx, resTensor.GetValue(idx) | offetInt9Bit_);
      }
    }
    if (tailLength != 0) {
      if (offsetLength_ > 1) {
        CalOffset(tailLength, eachLength * loops, resTensor);
      }
      DataCopyParams ub2GmParams1;
      SetCopyParams(ub2GmParams1);
      ub2GmParams1.blockLen = tailLength * sizeof(uint64_t);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      DataCopyGlobal(yGm_[eachLength * loops], resTensor, ub2GmParams1);
    }
  }

 protected:

  TPipe* pipe_;
  GlobalTensor<float> scaleGm_;
  GlobalTensor<float> offsetGm_;
  GlobalTensor<uint64_t> yGm_;

  TBuf<> resUb_;
  TBuf<> offsetUb_;

  uint32_t scaleLength_;
  uint32_t offsetLength_;

  uint64_t offetInt9Bit_ = 0;
  uint64_t scalequantPre_ = 0;

  template <typename T>
  __aicore__ inline void DataCopyLocal(const LocalTensor<T> &dst, const GlobalTensor<T> &src, DataCopyParams &params)
  {
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ > 200)
      DataCopyPadParams padParams;
      DataCopyPad(dst, src, params, padParams);
#else
      params.blockLen = Align<uint32_t>(params.blockLen, UB_ALIGN_SIZE) / UB_ALIGN_SIZE;
      DataCopy(dst, src, params);
#endif
  }

  template <typename T>
  __aicore__ inline void DataCopyGlobal(const GlobalTensor<T> &dst, const LocalTensor<T> &src, DataCopyParams &params)
  {
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ > 200)
      copy_ubuf_to_gm_align_b8((__gm__ void *)dst.GetPhyAddr(), (__ubuf__ void *)src.GetPhyAddr(), 0,
                               params.blockCount, params.blockLen, (uint8_t)0, (uint8_t)0, (uint32_t)0, (uint32_t)0);
#else
      params.blockLen = params.blockLen / sizeof(T);
      SafeDataCopy(dst, src, params.blockLen);
#endif
  }

  __aicore__ inline void CalQuantPreScale(uint32_t gmIdx, uint32_t idx, LocalTensor<uint64_t> resTensor) {
    if (scaleLength_ == 1) {
      resTensor.SetValue(idx, scalequantPre_);
      return;
    }
    uint64_t quantPre = 0;
    float scaleOri = scaleGm_.GetValue(gmIdx);
    uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&scaleOri));
    quantPre = (uint32Scale & DEQ_SCALE_MUL) | QUANT_SCALE;  // 取高19位
    resTensor.SetValue(idx, quantPre);
  }

  __aicore__ inline void CalOffsetValueShapeOne() {
    if (offsetLength_ != 1) {
      return;
    }
    uint32_t alignedLength = Align<uint32_t>(1 * sizeof(float), UB_ALIGN_SIZE) / sizeof(float);
    pipe_->InitBuffer(offsetUb_, 2 * alignedLength * sizeof(float)); // 需要2倍空间给 float和int32_t tensor
    LocalTensor<float> offsetFp32_ = offsetUb_.Get<float>(alignedLength);
    LocalTensor<int32_t> offsetInt32_ = offsetUb_.Get<int32_t>(alignedLength);
    DataCopyParams gm2UbParams;
    gm2UbParams.blockLen = 1 * sizeof(float);
    SetCopyParams(gm2UbParams);
    DataCopyLocal(offsetFp32_, offsetGm_, gm2UbParams);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    Cast(offsetInt32_, offsetFp32_, RoundMode::CAST_RINT, 1); // round to nearest, tie to even
    pipe_barrier(PIPE_V);
    Maxs(offsetInt32_, offsetInt32_, MIN_INT9, 1);
    pipe_barrier(PIPE_V);
    Mins(offsetInt32_, offsetInt32_, MAX_INT9, 1);
    pipe_barrier(PIPE_ALL);
    int offsetVal = offsetInt32_.GetValue(0);
    offetInt9Bit_ = (static_cast<uint64_t>(offsetVal) & QUANT_MASK_0) << OFFSET_DEVIATION;
  }

  __aicore__ inline void CalScaleValueShapeOne() {
    if (scaleLength_ != 1) {
      return;
    }
    float scaleOri = scaleGm_.GetValue(0);
    uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&scaleOri));
    scalequantPre_ = (uint32Scale & DEQ_SCALE_MUL) | QUANT_SCALE;
  }

  __aicore__ inline void SetOffsetValue(LocalTensor<uint64_t> resTensor, LocalTensor<int32_t> offsetInt32,
                                        LocalTensor<float> offsetFp32, uint32_t length) {
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    Cast(offsetInt32, offsetFp32, RoundMode::CAST_RINT, length); // round to nearest, tie to even
    pipe_barrier(PIPE_V);
    Maxs(offsetInt32, offsetInt32, MIN_INT9, length);
    pipe_barrier(PIPE_V);
    Mins(offsetInt32, offsetInt32, MAX_INT9, length);
    pipe_barrier(PIPE_ALL);
    for (uint32_t idx = 0; idx < length; ++idx) {
      int offsetVal = offsetInt32.GetValue(idx);
      uint64_t int9bits = (static_cast<uint64_t>(offsetVal) & QUANT_MASK_0) << OFFSET_DEVIATION;
      resTensor.SetValue(idx, resTensor.GetValue(idx) | int9bits);
    }
  }

  __aicore__ inline void CalOffset(uint32_t length, uint32_t gmOffset, LocalTensor<uint64_t> resTensor) {
    uint32_t alignedLength = Align<uint32_t>(length * sizeof(float), UB_ALIGN_SIZE) / sizeof(float);
    LocalTensor<float> offsetFp32_ = offsetUb_.Get<float>(alignedLength);
    LocalTensor<int32_t> offsetInt32_ = offsetUb_.Get<int32_t>(alignedLength);
    DataCopyParams gm2UbParams;
    gm2UbParams.blockLen = length * sizeof(float);
    SetCopyParams(gm2UbParams);
    DataCopyLocal(offsetFp32_, offsetGm_[gmOffset], gm2UbParams);
    SetOffsetValue(resTensor, offsetInt32_, offsetFp32_, length);
  }

  __aicore__ inline void SetCopyParams(DataCopyParams params) {
    params.blockCount = 1;
    params.srcStride = 0;
    params.dstStride = 0;
  }

  template <typename T>
  __aicore__ inline T Max(T a, T b) {
    return a > b ? a : b;
  }

  template <typename T>
  __aicore__ inline T Min(T a, T b) {
    return a > b ? b : a;
  }

  template <typename T>
  __aicore__ inline T Align(T a, T b = 16) {
    return (a + b - 1) / b * b;
  }

  __aicore__ inline int Float32ToInt9(float value) {
    int intValue = static_cast<int>(value);
    int int9Value = Max<int>(-256, Min<int>(255, intValue));
    return int9Value;
  }
};
}  // namespace AscendC

#endif  // TRANS_QUANT_PARAM_V2_H