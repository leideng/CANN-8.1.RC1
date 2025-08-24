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
 * \file reduce_common.h
 * \brief
 */
#ifndef _REDUCE_COMMON_H_
#define _REDUCE_COMMON_H_
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t MAX_REP_NUM = 255;
constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr uint32_t ELEM_PER_BLK_FP32 = 8;
constexpr float ZERO = 0;
constexpr int32_t HALf_INTERVAL = 2;

__aicore__ inline void ReduceSumForSmallReduceDimPreRepeat(const LocalTensor<float>& dstLocal,
                                                           const LocalTensor<float>& srcLocal,
                                                           const LocalTensor<float>& tmpLocal, const uint32_t elemNum,
                                                           const uint32_t numLastDim, const uint32_t tailCount,
                                                           const uint32_t repeat, const uint8_t repStride) {
  uint32_t elemIndex = 0;
  for (; elemIndex + ELEM_PER_REP_FP32 <= numLastDim; elemIndex += ELEM_PER_REP_FP32) {
    Add(tmpLocal, srcLocal[elemIndex], tmpLocal, elemNum, repeat,
        {1, 1, 1, ELEM_PER_BLK_FP32, repStride, ELEM_PER_BLK_FP32});
    pipe_barrier(PIPE_V);
  }
  if (unlikely(tailCount != 0)) {
    Add(tmpLocal, srcLocal[elemIndex], tmpLocal, tailCount, repeat,
        {1, 1, 1, ELEM_PER_BLK_FP32, repStride, ELEM_PER_BLK_FP32});
  }
  pipe_barrier(PIPE_V);
  AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);  // set mask = 64
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
  if ASCEND_IS_AIV {
    vcadd((__ubuf__ float*)dstLocal.GetPhyAddr(), (__ubuf__ float*)tmpLocal.GetPhyAddr(), repeat, 1, 1,
          ELEM_PER_BLK_FP32, false);
  }
#else
  vcadd((__ubuf__ float*)dstLocal.GetPhyAddr(), (__ubuf__ float*)tmpLocal.GetPhyAddr(), repeat, 1, 1,
        ELEM_PER_BLK_FP32);
#endif
}

/*
 * reduce dim form (N, D) to (N, 1)
 * this reduce sum is for small reduce dim.
 */
__aicore__ inline void ReduceSumForSmallReduceDim(const LocalTensor<float>& dstLocal,
                                                  const LocalTensor<float>& srcLocal,
                                                  const LocalTensor<float>& tmpLocal, const uint32_t numLastDimAligned,
                                                  const uint32_t numLastDim, const uint32_t tailCount,
                                                  const uint32_t repeat, const uint8_t repStride) {
  uint32_t repeatTimes = repeat / MAX_REP_NUM;
  if (repeatTimes == 0) {
    ReduceSumForSmallReduceDimPreRepeat(dstLocal, srcLocal, tmpLocal, ELEM_PER_REP_FP32, numLastDim, tailCount, repeat,
                                        repStride);
  } else {
    uint32_t repTailNum = repeat % MAX_REP_NUM;
    uint32_t repIndex = 0;
    uint32_t repElem;
    for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
      ReduceSumForSmallReduceDimPreRepeat(dstLocal[repIndex], srcLocal[repIndex * numLastDimAligned],
                                          tmpLocal[repIndex * ELEM_PER_REP_FP32], ELEM_PER_REP_FP32, numLastDim,
                                          tailCount, MAX_REP_NUM, repStride);
    }
    if (repTailNum != 0) {
      ReduceSumForSmallReduceDimPreRepeat(dstLocal[repIndex], srcLocal[repIndex * numLastDimAligned],
                                          tmpLocal[repIndex * ELEM_PER_REP_FP32], ELEM_PER_REP_FP32, numLastDim,
                                          tailCount, repTailNum, repStride);
    }
  }
}

/*
 * reduce dim form (N, D) to (N, 1)
 * this reduce sum is for small reduce dim, require D < 255 * 8.
 * size of tmpLocal: (N, 64)
 */
__aicore__ inline void ReduceSumMultiN(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal,
                                       const LocalTensor<float>& tmpLocal, const uint32_t numRow, const uint32_t numCol,
                                       const uint32_t numColAlign) {
  const uint32_t tailCount = numCol % ELEM_PER_REP_FP32;
  const uint32_t repeat = numRow;
  const uint8_t repStride = numColAlign / ELEM_PER_BLK_FP32;
  Duplicate(tmpLocal, ZERO, numRow * ELEM_PER_REP_FP32);
  pipe_barrier(PIPE_V);
  ReduceSumForSmallReduceDim(dstLocal, srcLocal, tmpLocal, numColAlign, numCol, tailCount, repeat, repStride);
}

__aicore__ inline int32_t findPowerTwo(int32_t n){
  // find max power of 2 no more than n (32 bit)
  n |= n >> 1;// Set the first digit of n's binary to 1
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return (n + 1) >> 1;
}

__aicore__ inline void ReduceSumHalfInterval(const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
                                             int32_t count) {
  if (likely(count > ELEM_PER_REP_FP32)) {
    int32_t bodyCount = findPowerTwo(count);
    int32_t tailCount = count - bodyCount;
    if (tailCount > 0) {
      Add(src_local, src_local, src_local[bodyCount], tailCount);
      pipe_barrier(PIPE_V);
    }
    while (bodyCount > ELEM_PER_REP_FP32) {
      bodyCount = bodyCount / HALf_INTERVAL;
      Add(src_local, src_local, src_local[bodyCount], bodyCount);
      pipe_barrier(PIPE_V);
    }

    AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
  } else {
    AscendCUtils::SetMask<float>(count);
  }
#if defined(__CCE_AICORE__) && __CCE_AICORE__  == 220
  if (g_coreType == AIV) {
    vcadd((__ubuf__ float*)dst_local.GetPhyAddr(), (__ubuf__ float*)src_local.GetPhyAddr(), 1, 0, 1, 0, false);
  }
#else
  vcadd((__ubuf__ float*)dst_local.GetPhyAddr(), (__ubuf__ float*)src_local.GetPhyAddr(), 1, 1, 1,
        DEFAULT_REPEAT_STRIDE);
#endif
  pipe_barrier(PIPE_V);
}

__aicore__ inline float ReduceSumHalfInterval(const LocalTensor<float>& src_local, int32_t count) {
  if (likely(count > ELEM_PER_REP_FP32)) {
    int32_t bodyCount = findPowerTwo(count);
    int32_t tailCount = count - bodyCount;
    if (tailCount > 0) {
      Add(src_local, src_local, src_local[bodyCount], tailCount);
      pipe_barrier(PIPE_V);
    }
    while (bodyCount > ELEM_PER_REP_FP32) {
      bodyCount = bodyCount / HALf_INTERVAL;
      Add(src_local, src_local, src_local[bodyCount], bodyCount);
      pipe_barrier(PIPE_V);
    }

    AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
  } else {
    AscendCUtils::SetMask<float>(count);
  }
#if defined(__CCE_AICORE__) && __CCE_AICORE__  == 220
  if (g_coreType == AIV) {
    vcadd((__ubuf__ float*)src_local.GetPhyAddr(), (__ubuf__ float*)src_local.GetPhyAddr(), 1, 0, 1, 0, false);
  }
#else
  vcadd((__ubuf__ float*)src_local.GetPhyAddr(), (__ubuf__ float*)src_local.GetPhyAddr(), 1, 1, 1,
        DEFAULT_REPEAT_STRIDE);
#endif
  event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
  set_flag(PIPE_V, PIPE_S, event_v_s);
  wait_flag(PIPE_V, PIPE_S, event_v_s);
  return src_local.GetValue(0);
}
#endif  // _REDUCE_COMMON_H_