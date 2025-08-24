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
 * \file dua_quantize_add_layer_norm_base.h
 * \brief
 */

#ifndef __DUA_QUANTIZE_ADD_LAYER_NORM_BASE_H_
#define __DUA_QUANTIZE_ADD_LAYER_NORM_BASE_H_

#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"

using namespace AscendC;
static constexpr float ZERO = 0;

#if __CCE_AICORE__ == 220
#define OUTPUT_MEAN_RSTD 1
#define SUPPORT_BF16 1
#else
#define OUTPUT_MEAN_RSTD 0
#define SUPPORT_BF16 0
#endif

template <typename Tp, Tp v>
struct integral_constant {
  static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

template <typename T, template <typename U> typename R, template <typename U> typename S>
__aicore__ inline void DataCopyEx(const R<T>& dst, const S<T>& src, const uint32_t len, const uint32_t count = 1,
                                  const DataCopyPadParams& padParams = {}) {
#if __CCE_AICORE__ == 220
  DataCopyParams copyParams;
  copyParams.blockLen = len * sizeof(T);
  copyParams.blockCount = count;
  if constexpr (is_same<R<T>, AscendC::LocalTensor<T>>::value) {
    DataCopyPad(dst, src, copyParams, padParams);
  } else {
    DataCopyPad(dst, src, copyParams);
  }
#else
  auto elementCount = len * count;
  int32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
  if (elementCount % numPerBlock == 0) {
    DataCopy(dst, src, elementCount);
  } else {
    if constexpr (is_same<R<T>, AscendC::LocalTensor<T>>::value) {
      auto num = AlignUp(elementCount, numPerBlock);
      DataCopy(dst, src, num);
    } else {
      int32_t num = elementCount / numPerBlock * numPerBlock;
      DataCopy(dst, src, num);
      if (elementCount != num) {
        event_t eventMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        set_flag(PIPE_MTE3, PIPE_S, eventMTE3S);
        wait_flag(PIPE_MTE3, PIPE_S, eventMTE3S);
        for (int32_t i = 0; i < numPerBlock; i++) {
          auto tensorValue = src.GetValue(elementCount - numPerBlock + i);
          src.SetValue(i, tensorValue);
        }
        event_t eventSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        set_flag(PIPE_S, PIPE_MTE3, eventSMTE3);
        wait_flag(PIPE_S, PIPE_MTE3, eventSMTE3);
        DataCopy(dst[elementCount - numPerBlock], src, numPerBlock);
      }
    }
  }
#endif
}

/*
 * only support count <= 255 * 64 = 16320
 */
__aicore__ inline float ReduceSumFP32(const LocalTensor<float>& src_local, int32_t count) {
  int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(float);
  int32_t repeatTimes = count / elementNumPerRep;
  int32_t tailCount = count % elementNumPerRep;
  int32_t bodyCount = repeatTimes * elementNumPerRep;
#ifdef __CCE_KT_TEST__
  assert(count <= MAX_REPEAT_TIMES * elementNumPerRep);
#endif
  float value = 0.0;
#if __CCE_AICORE__ == 220
  if (g_coreType == AIV) {
    if (likely(repeatTimes > 0)) {
      AscendCUtils::SetMask<float>(elementNumPerRep);
      vcadd(nullptr, (__ubuf__ float*)src_local.GetPhyAddr(), repeatTimes, 1, 1, 8, true);
      event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
      set_flag(PIPE_V, PIPE_S, eventVS);
      wait_flag(PIPE_V, PIPE_S, eventVS);
#ifdef __CCE_KT_TEST__
      uint64_t acc_val = get_acc_val();
#else
      uint64_t acc_val = GetAccVal();
#endif
      value = *reinterpret_cast<float*>(&acc_val);
    }
    if (unlikely(tailCount != 0)) {
      AscendCUtils::SetMask<float>(tailCount);
      vcadd(nullptr, (__ubuf__ float*)src_local[bodyCount].GetPhyAddr(), 1, 1, 1, 8, true);
      event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
      set_flag(PIPE_V, PIPE_S, eventVS);
      wait_flag(PIPE_V, PIPE_S, eventVS);
#ifdef __CCE_KT_TEST__
      uint64_t acc_val = get_acc_val();
#else
      uint64_t acc_val = GetAccVal();
#endif
      value += *reinterpret_cast<float*>(&acc_val);
    }
  }
#else
  ReduceSum(src_local, src_local, src_local, count);
  value = src_local.GetValue(0);
#endif
  return value;
}

__aicore__ inline void ReduceSumShort(const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
                                      const LocalTensor<float>& tmp_local, int32_t align_len, int32_t data_len,
                                      int32_t repeat) {
  int32_t elementNum = ONE_BLK_SIZE / sizeof(float);
  int32_t maxRepeat = ONE_REPEAT_BYTE_SIZE / sizeof(float);
  int32_t tailCount = data_len % elementNum;
  uint32_t index = 0;
  uint8_t repStride = align_len / ONE_BLK_FLOAT_NUM;

  int32_t repeatTimes = repeat / elementNum;
  int32_t bodyCount = repeatTimes * elementNum;
  int32_t repeatTail = repeat % elementNum * elementNum;

  Duplicate<float>(tmp_local, ZERO, repeat * elementNum);
  pipe_barrier(PIPE_V);
  for (index = 0; index + elementNum <= data_len; index += elementNum) {
    Add(tmp_local, tmp_local, src_local[index], elementNum, repeat, {1, 1, 1, 1, 1, repStride});
    pipe_barrier(PIPE_V);
  }
  if (unlikely(tailCount != 0)) {
    Add(tmp_local, tmp_local, src_local[index], tailCount, repeat, {1, 1, 1, 1, 1, repStride});
  }
  pipe_barrier(PIPE_V);
  if (repeatTimes != 0) {
    BlockReduceSum<float>(dst_local, tmp_local, repeatTimes, maxRepeat, 1, 1, elementNum);
  }
  if (repeatTail != 0) {
    BlockReduceSum<float>(dst_local[bodyCount], tmp_local[bodyCount * elementNum], 1, repeatTail, 1, 1, elementNum);
  }
}

#endif  // __DUA_QUANTIZE_ADD_LAYER_NORM_BASE_H_