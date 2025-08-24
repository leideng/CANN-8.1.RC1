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
 * \file reduce_xor_sum_common_impl.h
 * \brief
 */
#ifndef IMPL_REDUCE_REDUCE_XOR_SUM_REDUCE_XOR_SUM_COMMON_IMPL_H
#define IMPL_REDUCE_REDUCE_XOR_SUM_REDUCE_XOR_SUM_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_tiling/kernel_tiling.h"

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
#include "reduce_xor_sum_v220_impl.h"
#elif defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
#include "reduce_xor_sum_v200_impl.h"
#endif


#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220 || __CCE_AICORE__ == 200)
namespace AscendC {
namespace {
constexpr uint32_t REDUCE_XOR_SUM_REUSE_CALC_PROC = 2U;
constexpr uint32_t REDUCE_XOR_SUM_NOREUSE_CALC_PROC = 3U;
} // namespace


#if ASCENDC_CPU_DEBUG
template <typename T>
__aicore__ inline void ReduceXorSumCheckParams(LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                               const LocalTensor<T>& src1, const uint32_t calCount)
{
    ASCENDC_ASSERT(((calCount <= src0.GetSize()) && (calCount <= src1.GetSize())), {
        KERNEL_LOG(KERNEL_ERROR, "ReduceXorSum calCount is %u, which should "
            "not larger than src0Tensor size %u or src1Tensor size %u",
            calCount, src0.GetSize(), src1.GetSize());
    });

    ASCENDC_ASSERT((src0.GetSize() == src1.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "ReduceXorSum src0Tensor size %u must "
            "equal src1Tensor size %u.", src0.GetSize(), src1.GetSize());
    });

    constexpr uint32_t dstMinSize = ONE_BLK_SIZE / sizeof(T);
    ASCENDC_ASSERT((dst.GetSize() >= dstMinSize), {
        KERNEL_LOG(KERNEL_ERROR, "ReduceXorSum dstTensor size %u must be "
            "greater than minsize %u.", dst.GetSize(), dstMinSize);
    });
}
#endif

struct ReduceXorSumParam {
    __aicore__ ReduceXorSumParam() {};
    LocalTensor<int16_t> tmpTensor1;
    LocalTensor<int16_t> tmpTensor2;
    LocalTensor<int16_t> tmpTensor3;
};

#pragma begin_pipe(V)
template <typename T, bool isReuseSource = false>
__aicore__ inline void ReduceXorSumCompute(LocalTensor<T>& dst, const LocalTensor<T>& src0,
    const LocalTensor<T>& src1, LocalTensor<uint8_t>& tmp, const uint32_t calCount)
{
#if ASCENDC_CPU_DEBUG
    ReduceXorSumCheckParams(dst, src0, src1, calCount);
#endif

    uint32_t splitSize = 0;
    ReduceXorSumParam param;

    if constexpr (isReuseSource) {
        splitSize = tmp.GetSize() / sizeof(T) / REDUCE_XOR_SUM_REUSE_CALC_PROC / ONE_BLK_SIZE * ONE_BLK_SIZE;
        param.tmpTensor1 = tmp.ReinterpretCast<int16_t>();
        param.tmpTensor2 = param.tmpTensor1[splitSize];
        param.tmpTensor3 = src1;
    } else {
        splitSize = tmp.GetSize() / sizeof(T) / REDUCE_XOR_SUM_NOREUSE_CALC_PROC / ONE_BLK_SIZE * ONE_BLK_SIZE;
        param.tmpTensor1 = tmp.ReinterpretCast<int16_t>();
        param.tmpTensor2 = param.tmpTensor1[splitSize];
        param.tmpTensor3 = param.tmpTensor2[splitSize];
    }

    ASCENDC_ASSERT((splitSize >= calCount), {
        KERNEL_LOG(KERNEL_ERROR, "splitSize: %u must >= calCount: %u!", splitSize, calCount);
    });

    SetMaskCount();
    SetVectorMask<T>(0, calCount);
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    // x ^ y = (x | y) & (~(x & y))
    // (x & y)
    And<T, false>(param.tmpTensor1, src0, src1, MASK_PLACEHOLDER, 1, binaryParams);
    // (x | y)
    Or<T, false>(param.tmpTensor2, src0, src1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    // ~(x & y)
    Not<T, false>(param.tmpTensor1, param.tmpTensor1, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // z = (x | y) & (~(x & y)) <=> z = x ^ y
    And<T, false>(param.tmpTensor2, param.tmpTensor1, param.tmpTensor2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    CastInt162Float(param.tmpTensor1.ReinterpretCast<float>(), param.tmpTensor2);
    PipeBarrier<PIPE_V>();

    SetMaskNorm();
    ResetMask();

    ReduceSum<float>(param.tmpTensor1.ReinterpretCast<float>(), param.tmpTensor1.ReinterpretCast<float>(),
        param.tmpTensor3.ReinterpretCast<float>(), calCount);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<T>(0, 1);
    CastFloat2Int16(dst, param.tmpTensor1.ReinterpretCast<float>());
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}
#pragma end_pipe
}  // namespace AscendC

#endif

#endif  // IMPL_REDUCE_REDUCE_XOR_SUM_REDUCE_XOR_SUM_COMMON_IMPL_H
