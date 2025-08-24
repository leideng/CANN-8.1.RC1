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
 * \file rms_norm_grad_common.h
 * \brief
 */
#ifndef RMS_NORM_GRAD_BASE_H
#define RMS_NORM_GRAD_BASE_H
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#include "../norm_common/reduce_common.h"

using namespace AscendC;
static volatile __gm__ uint32_t fixed_output_sync[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM_DB = 2;
constexpr uint32_t FLOAT_DTYPE = 0;
constexpr uint32_t FLOAT16_DTYPE = 1;
constexpr uint32_t BFLOAT16_DTYPE = 2;
constexpr uint32_t ALIGN_32 = 8;
constexpr uint32_t ALIGN_16 = 16;
constexpr uint32_t CORE_NUM = 50;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t EACH_CORE_HANDLE_NUM = BLOCK_SIZE / sizeof(int32_t);
const int32_t CAL_ONE_BLOCK_FP32 = 8;
const uint32_t REDUCESUMTHRESHOLD = 64;
const uint32_t SMALLD_THRESHOLD = 640;
constexpr uint32_t ROW_FACTOR_SPLIT_D = 32;
constexpr int32_t DIM_NUM = 2;
constexpr int32_t DIM_N = 0;
constexpr int32_t DIM_D = 1;

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

__aicore__ inline void ReduceSumFP32(uint32_t idx, LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
                                     const LocalTensor<float>& work_local, int32_t count, uint32_t col_val_align) {
  if (g_coreType == AIV) {
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(float);
    uint64_t mask = elementNumPerRep;
    int32_t repeatTimes = count / elementNumPerRep;
    int32_t tailCount = count % elementNumPerRep;
    int32_t bodyCount = repeatTimes * elementNumPerRep;
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1RepStride = 0;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 0;
    repeatParams.dstBlkStride = 1;
    Duplicate(work_local, 0.0f, elementNumPerRep);
    pipe_barrier(PIPE_V);
    int32_t start_addr = idx * col_val_align;
    if (likely(repeatTimes > 0)) {
      Add(work_local, src_local[start_addr], work_local, mask, repeatTimes, repeatParams);
      pipe_barrier(PIPE_V);
    }
    if (unlikely(tailCount != 0)) {
      Add(work_local, src_local[start_addr + bodyCount], work_local, tailCount, 1, repeatParams);
      pipe_barrier(PIPE_V);
    }
    AscendCUtils::SetMask<float>(elementNumPerRep);
    vcadd((__ubuf__ float*)dst_local[start_addr].GetPhyAddr(), (__ubuf__ float*)work_local.GetPhyAddr(), 1, 0, 1, 0,
          false);
    pipe_barrier(PIPE_V);
  }
}

/*
 * only support count <= 255 * 64 = 16320
 */
__aicore__ inline float ReduceSumFP32_V2(const LocalTensor<float>& src_local, int32_t count) {
  int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(float);
  int32_t repeatTimes = count / elementNumPerRep;
  int32_t tailCount = count % elementNumPerRep;
  int32_t bodyCount = repeatTimes * elementNumPerRep;
#ifdef __CCE_KT_TEST__
  assert(count <= MAX_REPEAT_TIMES * elementNumPerRep);
#endif
  float value = 0.0;
  if (g_coreType == AIV) {
    if (likely(repeatTimes > 0)) {
      AscendCUtils::SetMask<float>(elementNumPerRep);
      vcadd(nullptr, (__ubuf__ float*)src_local.GetPhyAddr(), repeatTimes, 1, 1, 8, true);
      event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
      set_flag(PIPE_V, PIPE_S, eventId);
      wait_flag(PIPE_V, PIPE_S, eventId);
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
      event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
      set_flag(PIPE_V, PIPE_S, eventId);
      wait_flag(PIPE_V, PIPE_S, eventId);
#ifdef __CCE_KT_TEST__
      uint64_t acc_val = get_acc_val();
#else
      uint64_t acc_val = GetAccVal();
#endif
      value += *reinterpret_cast<float*>(&acc_val);
    }
  }
  return value;
}

template <typename T>
__aicore__ inline void Cast2FloatIf(LocalTensor<float>& castLocal, uint32_t srcOffset, uint32_t calcCount) {
  if constexpr (!is_same<T, float>::value) {
    LocalTensor<T> castLocalB16 = castLocal.ReinterpretCast<T>();
    Cast(castLocal, castLocalB16[srcOffset], RoundMode::CAST_NONE, calcCount);
    pipe_barrier(PIPE_V);
  }
}

__aicore__ inline uint32_t ROUND_UP(uint32_t x, uint32_t block_number) {
  if (block_number > 0) {
    return (x + block_number - 1) / block_number * block_number;
  }
  return 0;
}

template <typename T>
__aicore__ inline void DataCopyCustom(GlobalTensor<T> &dstTensor, LocalTensor<T> &srcTensor, const uint32_t dstOffset,
const uint32_t srcOffset, const uint32_t blockCount) {
  uint32_t alignLen_= ALIGN_32;
  if constexpr (!is_same<T, float>::value) {
    alignLen_= ALIGN_16;
  }
  uint32_t calcLenAlign32 = (blockCount / alignLen_) * alignLen_;
  if(calcLenAlign32 > 0){
      DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], calcLenAlign32);
  }
  uint32_t calcLenTail32 = blockCount % alignLen_;
  
  if (calcLenTail32 > 0) {
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      for (uint32_t i = 0; i < calcLenTail32; i++) {
          dstTensor.SetValue(dstOffset + calcLenAlign32 + i, srcTensor.GetValue(srcOffset + calcLenAlign32 + i));
      }
      DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(dstTensor);
      pipe_barrier(PIPE_ALL);
      set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  }
}

template <typename T>
__aicore__ inline void InitGmZero(GlobalTensor<T> &outGm, TBuf<TPosition::VECCALC> &TmpZeroTBuf, 
const uint32_t zeroLen, const uint32_t outOffset) {
  uint32_t alignLen_ = BLOCK_SIZE / sizeof(T);
  LocalTensor<T> temp_zero_tensor = TmpZeroTBuf.Get<T>();
  
  Duplicate(temp_zero_tensor, (T)0.0, zeroLen);
  pipe_barrier(PIPE_ALL);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

  DataCopy(outGm[outOffset], temp_zero_tensor, ROUND_UP(zeroLen, alignLen_));
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  
  pipe_barrier(PIPE_ALL);
}
#endif  // RMS_NORM_GRAD_BASE_H