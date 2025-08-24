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
 * \file xor_membase_impl.h
 * \brief
 */
#ifndef IMPL_MATH_XOR_XOR_MEMBASE_IMPL_H
#define IMPL_MATH_XOR_XOR_MEMBASE_IMPL_H
#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
template <typename T>
__aicore__ inline void XorCalcSimplified(const LocalTensor<T>& dstAddr, const LocalTensor<T> &src0Addr,
    const LocalTensor<T> &src1Addr, const LocalTensor<T>& tmpTensor)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    // (x & y)
    And<T, false>(dstAddr, src0Addr, src1Addr, MASK_PLACEHOLDER, 1, binaryParams);
    // (x | y)
    Or<T, false>(tmpTensor, src0Addr, src1Addr, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    // ~(x & y)
    Not<T, false>(dstAddr, dstAddr, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // z = (x | y) & (~(x & y))
    And<T, false>(dstAddr, tmpTensor, dstAddr, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}
template <typename T, bool isReuseSource = false>
__aicore__ inline void XorImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T> &src0Tensor,
    const LocalTensor<T> &src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
#if defined(ASCENDC_CPU_DEBUG) && (ASCENDC_CPU_DEBUG == 1)
    IsXorParamValid(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
#endif

    uint32_t stackSize = sharedTmpBuffer.GetSize() / sizeof(T) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    const uint32_t loopCount = calCount / stackSize;
    const uint32_t tail = calCount % stackSize;

    SetMaskCount();
    SetVectorMask<T>(0, stackSize);
    for (uint32_t i = 0; i < loopCount; i++) {
        XorCalcSimplified(dstTensor[i * stackSize],
            src0Tensor[i * stackSize],
            src1Tensor[i * stackSize],
            sharedTmpBuffer.ReinterpretCast<T>());
    }
    if (tail != 0) {
        SetVectorMask<T>(0, tail);
        XorCalcSimplified(dstTensor[loopCount * stackSize],
            src0Tensor[loopCount * stackSize],
            src1Tensor[loopCount * stackSize],
            sharedTmpBuffer.ReinterpretCast<T>());
    }
    SetMaskNorm();
    SetVectorMask<T>(FULL_MASK, FULL_MASK);
}
} // AscendC
#endif  // IMPL_MATH_XOR_XOR_MEMBASE_IMPL_H
