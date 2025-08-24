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
 * \file utils.h
 * \brief
 */
#ifndef UTILS_H
#define UTILS_H
#include "kernel_operator.h"

__aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y = 16)
{
  if (y == 0) {
    return x;
  }
  return (x + y - 1) / y * y;
}

template <typename T>
__aicore__ inline void CastFrom32To16(const AscendC::LocalTensor<T> &out, const AscendC::LocalTensor<float> &in,
    uint32_t count)
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(out, in, AscendC::RoundMode::CAST_NONE, count); // 310p cast fp32->half 只能用CAST_NONE，这里拉齐310p和910b
    } else { // bf16
        Cast(out, in, AscendC::RoundMode::CAST_RINT, count);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void CastFromF16ToI8(const AscendC::LocalTensor<int8_t> &out, const AscendC::LocalTensor<half> &in,
    half quantMin, uint32_t count)
{
    Maxs(in, in, quantMin, count);
    AscendC::PipeBarrier<PIPE_V>();
    Mins(in, in, (half)127, count); // 127: limit
    AscendC::PipeBarrier<PIPE_V>();
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    Cast(out, in, AscendC::RoundMode::CAST_RINT, count);
#else
    Cast(out, in, AscendC::RoundMode::CAST_NONE, count);
#endif
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void ComputeMean(const AscendC::LocalTensor<T> &out, const AscendC::LocalTensor<T> &in,
    T aveNum, uint32_t count)
{
    Duplicate(out, aveNum, count);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(out, in, out, count);
    AscendC::PipeBarrier<PIPE_V>();
    T sum = ComputeSum(out, out, out, count);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    Duplicate(out, sum, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void CopyGmTilingToUb(__ubuf__ uint8_t *tilingInUb, const __gm__ uint8_t *tilingInGm,
                                        size_t tilingSize, AscendC::TPipe *pipe)
{
    uint32_t roundTilingSize = RoundUp(tilingSize, 32);
    AscendC::TBuf<AscendC::TPosition::VECCALC> tilingBuf;
    AscendC::GlobalTensor<uint8_t> tilingGm;

    tilingGm.SetGlobalBuffer((__gm__ uint8_t *)tilingInGm);
    pipe->InitBuffer(tilingBuf, roundTilingSize);

    AscendC::LocalTensor<uint8_t> tilingUb = tilingBuf.Get<uint8_t>();
    AscendC::DataCopy(tilingUb, tilingGm, roundTilingSize);

    tilingInUb = (__ubuf__ uint8_t *)tilingUb.GetPhyAddr();
}
#endif