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
 * \file sum_common_impl.h
 * \brief
 */
#ifndef IMPL_REDUCE_SUM_SUM_COMMON_IMPL_H
#define IMPL_REDUCE_SUM_SUM_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_log.h"
#include <type_traits>
#endif
namespace AscendC {
struct SumParams {
    uint32_t outter = 1;
    uint32_t inner;  // inner = 32-byte alignment of n, inner = (n *sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T)
    uint32_t n;      // actual length of the tensor
};

template <typename T>
__aicore__ inline void CheckParamsIsValid(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const SumParams &sumParams)
{
#if ASCENDC_CPU_DEBUG
    bool ans = dstTensor.GetSize() > 0 && srcTensor.GetSize() > 0 && sharedTmpBuffer.GetSize() > 0;
    ASCENDC_ASSERT(ans, { KERNEL_LOG(KERNEL_ERROR, "LocalTensor size must be greater than 0!"); });
    ans = sumParams.inner != 0 && (sumParams.inner * sizeof(T) % ONE_BLK_SIZE == 0);
    ASCENDC_ASSERT(ans, { KERNEL_LOG(KERNEL_ERROR, "inner must be 32B aligned"); });
    ans = (sumParams.inner <= srcTensor.GetSize());
    ASCENDC_ASSERT(ans, { KERNEL_LOG(KERNEL_ERROR, "inner must be less than or equal to src tensor"); });
    ans = ((sumParams.n >= 1) && (sumParams.n <= sumParams.inner));
    ASCENDC_ASSERT(
        ans, { KERNEL_LOG(KERNEL_ERROR, "n must be greater than or equal to 1 and less than or equal to inner"); });
    ans = dstTensor.GetSize() * sizeof(T) >=
          (sumParams.outter * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    ASCENDC_ASSERT(ans, { KERNEL_LOG(KERNEL_ERROR, "dstTensor size isn't enough!"); });
    ans = (std::is_same<T, half>::value) || (std::is_same<T, float>::value);
    ASCENDC_ASSERT(ans, { KERNEL_LOG(KERNEL_ERROR, "type must be half or float"); });
#endif
}

template <typename T>
__aicore__ inline void SumForOneRepeatTime(
    const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const SumParams &sumParams)
{
    SetVectorMask<T>(0, sumParams.n);
    for (uint32_t row = 0; row < sumParams.outter; ++row) {
        RepeatReduceSum<T, false>(dstTensor[row], srcTensor[row * sumParams.inner], 1, MASK_PLACEHOLDER,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    }
    SetMaskNorm();
    ResetMask();
}

template <typename T, int32_t reduceDim = -1, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SumCompute(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const SumParams &sumParams)
{
    if ASCEND_IS_AIC {
        return;
    }
    CheckParamsIsValid(dstTensor, srcTensor, sharedTmpBuffer, sumParams);
#if __CCE_AICORE__ >= 200
    uint32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
    uint32_t firstRepeatTimes = (sumParams.n + elementNumPerRep - 1) / elementNumPerRep;
    SetMaskCount();
    if (firstRepeatTimes == 1) {
        return SumForOneRepeatTime(dstTensor, srcTensor, sumParams);
    }
    uint32_t totalCnt = 1;
    uint32_t dataSize = firstRepeatTimes;
    while (dataSize > 1) {
        ++totalCnt;
        dataSize = (dataSize + elementNumPerRep - 1) / elementNumPerRep;
    }
    LocalTensor<T> tmpTensor = sharedTmpBuffer.ReinterpretCast<T>();
    for (uint32_t row = 0; row < sumParams.outter; ++row) {
        uint32_t cnt = totalCnt;
        uint64_t lowMask = sumParams.n;
        SetVectorMask<T>(0, lowMask);
        RepeatReduceSum<T, false>(tmpTensor, srcTensor[row * sumParams.inner], 1, MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);

        PipeBarrier<PIPE_V>();
        lowMask = (lowMask + elementNumPerRep - 1) / elementNumPerRep;
        --cnt;
        while (cnt != 0) {
            SetVectorMask<T>(0, lowMask);
            if (cnt == 1) {
                RepeatReduceSum<T, false>(dstTensor[row], tmpTensor, 1, MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE,
                    DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
            } else {
                RepeatReduceSum<T, false>(tmpTensor, tmpTensor, 1, MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE,
                    DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
            }
            PipeBarrier<PIPE_V>();
            lowMask = (lowMask + elementNumPerRep - 1) / elementNumPerRep;
            --cnt;
        }
    }
    SetMaskNorm();
#endif
}

}  // namespace AscendC
#endif // IMPL_REDUCE_SUM_SUM_COMMON_IMPL_H
