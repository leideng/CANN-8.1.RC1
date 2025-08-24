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
using AscendC::HardEvent;
__aicore__ inline uint32_t Min(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
__aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y = 16)
{
  if (y == 0) {
    return x;
  }
    return (x + y - 1) / y * y;
}
__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : ((x + y - 1) / y);
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
template <typename T>
__aicore__ inline void CastFrom16To32(const AscendC::LocalTensor<float> &out, const AscendC::LocalTensor<T> &in,
    uint32_t count)
{
    Cast(out, in, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, typename Q>
__aicore__ inline void CopyIn(const AscendC::GlobalTensor<T> &gm, Q &queue, uint64_t offset, uint32_t count)
{
    AscendC::LocalTensor<T> local = queue.template AllocTensor<T>();
    DataCopy(local, gm[offset], count);
    queue.EnQue(local);
}
template <typename T, typename Q>
__aicore__ inline void CopyOut(const AscendC::GlobalTensor<T> &gm, Q &queue, uint64_t offset, uint32_t count)
{
    AscendC::LocalTensor<T> local = queue.template DeQue<T>();
    DataCopy(gm[offset], local, count);
    queue.FreeTensor(local);
}
template <typename T, typename Q>
__aicore__ inline void CopyInAndCastF32(const AscendC::LocalTensor<float> &out, const AscendC::GlobalTensor<T> &gm,
    Q &queue, uint64_t offset, uint32_t count)
{
    CopyIn(gm, queue, offset, count);
    AscendC::LocalTensor<T> local = queue.template DeQue<T>();
    Cast(out, local, AscendC::RoundMode::CAST_NONE, count);
    queue.FreeTensor(local);
    AscendC::PipeBarrier<PIPE_V>();
}
template <typename T>
__aicore__ inline T ComputeSum(const AscendC::LocalTensor<T> &in, const AscendC::LocalTensor<T> &tmp,
                               const AscendC::LocalTensor<T> &workLocal, uint32_t count)
{
#if __CCE_AICORE__ == 100
    float sum = 0;
    int64_t elementNumPerRep = AscendC::ONE_REPEAT_BYTE_SIZE / sizeof(T);
    AscendC::LocalTensor<T> src = in;
    while (count > elementNumPerRep) {
        int64_t repeatTimes = count / elementNumPerRep;
        int64_t tailCount = count % elementNumPerRep;
        int64_t bodyCount = repeatTimes * elementNumPerRep;
        if (repeatTimes > 0) {
            AscendC::AscendCUtils::SetMask<T>(elementNumPerRep);
            vcadd((__ubuf__ T *)tmp.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), repeatTimes, 1, 1, 8);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0); // PipeBarrier(PIPE_V)?
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }

        if (tailCount != 0) {
            AscendC::AscendCUtils::SetMask<T>(tailCount);
            vcadd((__ubuf__ T *)tmp[bodyCount].GetPhyAddr(), (__ubuf__ T *)src[bodyCount].GetPhyAddr(), 1, 1, 1, 8);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            sum += tmp.GetValue(bodyCount);
        }

        count = repeatTimes;
        src = tmp;
    }

    if (count > 1) {
        AscendC::AscendCUtils::SetMask<T>(count);
        vcadd((__ubuf__ T *)tmp.GetPhyAddr(), (__ubuf__ T *)tmp.GetPhyAddr(), 1, 1, 1, 8);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    sum += tmp.GetValue(0);
    return sum;
#else
    ReduceSum(tmp, in, workLocal, count);
    AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
    return tmp.GetValue(0);
#endif
}
template <typename T>
__aicore__ inline void ComputeResidualAdd(const AscendC::LocalTensor<T> &out,
    const AscendC::LocalTensor<T> &in, const AscendC::LocalTensor<T> &resIn, uint32_t count)
{
    Add(out, in, resIn, count);
    AscendC::PipeBarrier<PIPE_V>();
}
__aicore__ inline float ComputeSliceSquareSum(const AscendC::LocalTensor<float> &in,
    const AscendC::LocalTensor<float> &tmp, const AscendC::LocalTensor<float> &workLocal, uint32_t count)
{
    Mul(tmp, in, in, count);
    AscendC::PipeBarrier<PIPE_V>();
    return ComputeSum(tmp, tmp, workLocal, count);
}

template <typename T, typename Q>
__aicore__ inline void Cast16AndCopyOut(const AscendC::LocalTensor<float> &in, const AscendC::GlobalTensor<T> &gm,
    Q &queue, uint64_t offset, uint32_t count)
{
    AscendC::LocalTensor<T> local = queue.template AllocTensor<T>();
    CastFrom32To16(local, in, count);
    queue.EnQue(local);
    CopyOut(gm, queue, offset, count);
    AscendC::PipeBarrier<PIPE_V>();
}
template <typename T>
__aicore__ inline void ComputeLayerNorm(const AscendC::LocalTensor<float> &out, const AscendC::LocalTensor<float> &in,
    const AscendC::LocalTensor<float> &mean, float eps, float aveNum, const AscendC::LocalTensor<T> &gamma,
    const AscendC::LocalTensor<T> &beta, uint32_t count)
{
    Sub(in, in, mean, count);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(out, in, in, count);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(out, out, aveNum, count);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceSum(out, out, out, count);
    AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float var = out.GetValue(0);
    AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(out, var, count);
    AscendC::PipeBarrier<PIPE_V>();
    Adds(out, out, eps, count);
    AscendC::PipeBarrier<PIPE_V>();
    Sqrt(out, out, count);
    AscendC::PipeBarrier<PIPE_V>();

    Div(out, in, out, count);
    AscendC::PipeBarrier<PIPE_V>();

    Cast(in, gamma, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(out, out, in, count);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(in, beta, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    Add(out, out, in, count);
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