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
 * \file add_rms_norm_base.h
 * \brief
 */
#ifndef _ADD_RMS_NORM_BASE_H_
#define _ADD_RMS_NORM_BASE_H_
#include "kernel_operator.h"

using namespace AscendC;

#if __CCE_AICORE__ != 220
#define bfloat16_t int16_t
#endif
constexpr int32_t BUFFER_NUM = 1;         // tensor num for each queue
constexpr int32_t NUM_PER_REP_FP32 = 64;  // ONE_REPEAT_BYTE_SIZE / sizeof(float);
constexpr int32_t NUM_PER_BLK_FP32 = 8;
constexpr float MINUS_HALF = -0.5;
constexpr float ZERO = 0;
constexpr float ONE = 1;

template <typename T>
__aicore__ inline T CeilDiv(T x, T y) {
  return y == 0 ? x : (x + y - 1) / y;
}

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

__aicore__ inline void ReduceSumFP32(const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
                                     const LocalTensor<float>& work_local, int32_t count) {
  // count need smaller than 255 repeat
  if (g_coreType == AIV) {
    uint64_t mask = NUM_PER_REP_FP32;
    int32_t repeatTimes = count / NUM_PER_REP_FP32;
    int32_t tailCount = count % NUM_PER_REP_FP32;
    int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1RepStride = 0;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 0;
    repeatParams.dstBlkStride = 1;
    Duplicate(work_local, ZERO, NUM_PER_REP_FP32);
    pipe_barrier(PIPE_V);
    if (likely(repeatTimes > 0)) {
      Add(work_local, src_local, work_local, mask, repeatTimes, repeatParams);
      pipe_barrier(PIPE_V);
    }
    if (unlikely(tailCount != 0)) {
      Add(work_local, src_local[bodyCount], work_local, tailCount, 1, repeatParams);
      pipe_barrier(PIPE_V);
    }
    AscendCUtils::SetMask<float>(NUM_PER_REP_FP32);
    vcadd((__ubuf__ float*)dst_local.GetPhyAddr(), (__ubuf__ float*)work_local.GetPhyAddr(), 1, 0, 1, 0, false);
    pipe_barrier(PIPE_V);
  }
}

__aicore__ inline void ReduceSumCustom(const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
                                       const LocalTensor<float>& work_local, int32_t count) {
#if __CCE_AICORE__ == 220
  ReduceSumFP32(dst_local, src_local, work_local, count);
#else
  ReduceSum(dst_local, src_local, dst_local, count);
#endif
}

__aicore__ inline void ReduceSumFP32ToBlock(const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
                                            const LocalTensor<float>& work_local, int32_t count) {
  // count need smaller than 255 repeat
  uint64_t mask = NUM_PER_REP_FP32;
  int32_t repeatTimes = count / NUM_PER_REP_FP32;
  int32_t tailCount = count % NUM_PER_REP_FP32;
  int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
  BinaryRepeatParams repeatParams;
  repeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
  repeatParams.src0BlkStride = 1;
  repeatParams.src1RepStride = 0;
  repeatParams.src1BlkStride = 1;
  repeatParams.dstRepStride = 0;
  repeatParams.dstBlkStride = 1;
  Duplicate(work_local, ZERO, NUM_PER_REP_FP32);
  pipe_barrier(PIPE_V);
  if (likely(repeatTimes > 0)) {
    Add(work_local, src_local, work_local, mask, repeatTimes, repeatParams);
    pipe_barrier(PIPE_V);
  }
  if (unlikely(tailCount != 0)) {
    Add(work_local, src_local[bodyCount], work_local, tailCount, 1, repeatParams);
    pipe_barrier(PIPE_V);
  }
  BlockReduceSum(dst_local, work_local, 1, mask, 1, 1, DEFAULT_REPEAT_STRIDE);
  pipe_barrier(PIPE_V);
}

__aicore__ inline void BlockReduceSumFP32(const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
                                          int32_t count) {
  // count need multiple of 8
  int32_t repeatTimes = count / NUM_PER_REP_FP32;
  int32_t tailCount = count % NUM_PER_REP_FP32;
  int32_t dstAddr = repeatTimes * 8;
  int32_t srcAddr = repeatTimes * NUM_PER_REP_FP32;
  if (likely(repeatTimes > 0)) {
    BlockReduceSum(dst_local, src_local, repeatTimes, NUM_PER_REP_FP32, 1, 1, DEFAULT_REPEAT_STRIDE);
    pipe_barrier(PIPE_V);
  }
  if (tailCount != 0) {
    BlockReduceSum(dst_local[dstAddr], src_local[srcAddr], 1, tailCount, 1, 1, DEFAULT_REPEAT_STRIDE);
    pipe_barrier(PIPE_V);
  }
}

template <typename T, typename U, typename R>
__aicore__ inline void DataCopyCustom(const U& dstTensor, const R& srcTensor, const uint32_t count) {
#if __CCE_AICORE__ == 220
  DataCopyParams copyParams;
  copyParams.blockLen = count * sizeof(T);
  copyParams.blockCount = 1;
  if constexpr (is_same<U, AscendC::LocalTensor<T>>::value) {
    DataCopyPadParams padParams;
    DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
  } else {
    DataCopyPad(dstTensor, srcTensor, copyParams);
  }
#else
  // only support count greater than 32byte
  int32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
  if (count % numPerBlock == 0) {
    DataCopy(dstTensor, srcTensor, count);
  } else {
    if constexpr (is_same<U, AscendC::LocalTensor<T>>::value) {
      int32_t num = AlignUp(count, numPerBlock);
      DataCopy(dstTensor, srcTensor, num);
    } else {
      int32_t num = count / numPerBlock * numPerBlock;
      DataCopy(dstTensor, srcTensor, num);
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      for (int32_t i = 0; i < numPerBlock; i++) {
        T tensorValue = srcTensor.GetValue(count - numPerBlock + i);
        srcTensor.SetValue(i, tensorValue);
      }
      set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      DataCopy(dstTensor[count - numPerBlock], srcTensor, numPerBlock);
    }
  }
#endif
}
#endif  // ADD_RMS_NORM_BASE_H_