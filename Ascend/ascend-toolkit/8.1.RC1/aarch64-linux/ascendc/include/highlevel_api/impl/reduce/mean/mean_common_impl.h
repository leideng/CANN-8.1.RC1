/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mean_common_impl.h
 * \brief
 */
#ifndef LIB_REDUCE_MEAN_MEAN_COMMON_IMPL_H
#define LIB_REDUCE_MEAN_MEAN_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_log.h"
#include <type_traits>
#endif

namespace AscendC {
constexpr uint32_t HALF_NUM_PER = 128;
constexpr uint32_t FLOAT_NUM_PER = 64;
struct MeanParams {
    uint32_t outter = 1;
    uint32_t inner;  // inner = 32-byte alignment of n, inner = (n *sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T)
    uint32_t n;      // actual length of the tensor
};

template <typename T>
__aicore__ inline void CheckParamsIsValid(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const MeanParams& meanParams, uint32_t tmpBufferSize)
{
#if ASCENDC_CPU_DEBUG
    bool ans = meanParams.outter != 0 && meanParams.inner != 0 && (meanParams.inner * sizeof(T) % ONE_BLK_SIZE == 0);
    ASCENDC_ASSERT(ans, { KERNEL_LOG(KERNEL_ERROR, "outter and inner can't be zero, inner must be 32B aligned"); });
    ans = ((meanParams.n >= 1) && (meanParams.n <= meanParams.inner));
    ASCENDC_ASSERT(
        ans, { KERNEL_LOG(KERNEL_ERROR, "n must be greater than or equal to 1 and less than or equal to inner"); });
    ans = srcTensor.GetSize() >= meanParams.outter * meanParams.inner;
    ASCENDC_ASSERT(ans, { KERNEL_LOG(KERNEL_ERROR, "srcTensor size isn't enough!"); });
    ans = dstTensor.GetSize() * sizeof(T) >=
          (meanParams.outter * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    ASCENDC_ASSERT(ans, { KERNEL_LOG(KERNEL_ERROR, "dstTensor size isn't enough!"); });
    ans = sharedTmpBuffer.GetSize() >= tmpBufferSize;
    ASCENDC_ASSERT(ans, { KERNEL_LOG(KERNEL_ERROR, "sharedTmpBuffer size isn't enough!"); });
#endif
}

__aicore__ inline void MeanCast(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const MeanParams& meanParams)
{
    uint32_t elementNumPerRep = FLOAT_NUM_PER;
    uint32_t repeateTimes = (meanParams.n + elementNumPerRep - 1) / elementNumPerRep;
    uint32_t finalWorkSize =
        meanParams.inner * sizeof(float) + (repeateTimes + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
#if ASCENDC_CPU_DEBUG
    CheckParamsIsValid(dstTensor, srcTensor, sharedTmpBuffer, meanParams, finalWorkSize);
#endif
    const UnaryRepeatParams unaryParams;
    float scalarValue = static_cast<float>(1) / static_cast<float>(static_cast<int32_t>(meanParams.n));
    LocalTensor<float> TmpTensor = sharedTmpBuffer.ReinterpretCast<float>();
    LocalTensor<half> castTensor = sharedTmpBuffer.ReinterpretCast<half>();
    SetMaskCount();
    for (uint32_t row = 0; row < meanParams.outter; ++row) {
        SetVectorMask<half>(0, meanParams.n);
        Cast<float, half, false>(TmpTensor, srcTensor[row * meanParams.inner], RoundMode::CAST_NONE, MASK_PLACEHOLDER,
                                 1, {1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
        RepeatReduceSum<float, false>(TmpTensor[meanParams.inner], TmpTensor, 1,
                                      MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE,
                                      DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        uint32_t reduceNums = repeateTimes;
        while (reduceNums > 1) {
            SetVectorMask<half>(0, reduceNums);
            reduceNums = (reduceNums + elementNumPerRep - 1) / elementNumPerRep;
            RepeatReduceSum<float, false>(TmpTensor[meanParams.inner], TmpTensor[meanParams.inner], 1,
                                      MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE,
                                      DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);

            PipeBarrier<PIPE_V>();
        }
        SetVectorMask<half>(0, 1);
        Muls<float, false>(TmpTensor[meanParams.inner], TmpTensor[meanParams.inner],
                       scalarValue, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(castTensor, TmpTensor[meanParams.inner], RoundMode::CAST_NONE, MASK_PLACEHOLDER,
                                 1, {1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
        RepeatReduceSum<half, false>(dstTensor[row], castTensor, 1, MASK_PLACEHOLDER,
                                     DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    }
    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void MeanForOneRepeatTime(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                            const MeanParams& meanParams, T scalarValue)
{
    SetVectorMask<T>(0, meanParams.n);
    for (uint32_t row = 0; row < meanParams.outter; ++row) {
        RepeatReduceSum<T, false>(dstTensor[row], srcTensor[row * meanParams.inner], 1, MASK_PLACEHOLDER,
                                  DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    }
    PipeBarrier<PIPE_V>();
    SetVectorMask<T>(0, meanParams.outter);
    const UnaryRepeatParams unaryParams;
    Muls<T, false>(dstTensor, dstTensor, scalarValue, MASK_PLACEHOLDER, 1, unaryParams);
    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void MeanCommon(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const MeanParams& meanParams)
{
    uint32_t elementNumPerRep = FLOAT_NUM_PER;
    if constexpr (sizeof(T) == sizeof(half)) {
        elementNumPerRep = HALF_NUM_PER;
    }
    uint32_t repeateTimes = (meanParams.n + elementNumPerRep - 1) / elementNumPerRep;
    uint32_t finalWorkSize = (repeateTimes + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
#if ASCENDC_CPU_DEBUG
    CheckParamsIsValid(dstTensor, srcTensor, sharedTmpBuffer, meanParams, finalWorkSize);
#endif
    T scalarValue = static_cast<T>(static_cast<float>(1) / static_cast<float>(static_cast<int32_t>(meanParams.n)));
    SetMaskCount();
    if (repeateTimes == 1) {
        return MeanForOneRepeatTime(dstTensor, srcTensor, meanParams, scalarValue);
    }
    const UnaryRepeatParams unaryParams;
    LocalTensor<T> TmpTensor = sharedTmpBuffer.ReinterpretCast<T>();
    for (uint32_t row = 0; row < meanParams.outter; ++row) {
        uint32_t reduceNums = repeateTimes;
        SetVectorMask<T>(0, meanParams.n);
        RepeatReduceSum<T, false>(TmpTensor,
            srcTensor[row * meanParams.inner],
            1,
            MASK_PLACEHOLDER,
            DEFAULT_BLK_STRIDE,
            DEFAULT_BLK_STRIDE,
            DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        while (reduceNums > 1) {
            SetVectorMask<T>(0, reduceNums);
            reduceNums = (reduceNums + elementNumPerRep - 1) / elementNumPerRep;
            if (reduceNums == 1) {
                RepeatReduceSum<T, false>(dstTensor[row], TmpTensor, 1, MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE,
                                          DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
            } else {
                RepeatReduceSum<T, false>(TmpTensor, TmpTensor, 1, MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE,
                                          DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
            }
            PipeBarrier<PIPE_V>();
        }
    }
    SetVectorMask<T>(0, meanParams.outter);
    Muls<T, false>(dstTensor, dstTensor, scalarValue, MASK_PLACEHOLDER, 1, unaryParams);
    SetMaskNorm();
}

#pragma end_pipe
}  // namespace AscendC

#endif  // LIB_REDUCE_MEAN_H