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
 * \file reduce_xor_sum.h
 * \brief
 */
#ifndef LIB_REDUCE_REDUCE_XOR_SUM_H
#define LIB_REDUCE_REDUCE_XOR_SUM_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "../../impl/reduce/reduce_xor_sum/reduce_xor_sum_common_impl.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_log.h"
#include <type_traits>
#endif

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ >= 220 || __CCE_AICORE__ == 200)

namespace AscendC {
#pragma begin_pipe(V)
/*
 * @ingroup ReduceXorSum
 * @brief f(x) = sum(a ^ b)
 *        If the final calculation result is beyond the range of int16,
 *        the calculation result is not guaranteed.
 * @tparam T: Input and output data types, int16
 * @tparam isReuseSrc: Whether temporary variables can reuse the input memory.
 * @param [out] dstTensor: output LocalTensor, the minimum shape is 16.
 * @param [in] src0Tensor: input0 LocalTensor
 * @param [in] src1Tensor: input1 LocalTensor
 * @param [in] sharedTmpBuffer: input local temporary Tensor
 * @param [in] calCount: amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void ReduceXorSum(LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(std::is_same<T, int16_t>::value, "ReduceXorSum only support int16_t data type on current device!");
    ReduceXorSumCompute<T, isReuseSource>(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
}

/*
 * @ingroup ReduceXorSum
 * @brief f(x) = sum(a ^ b)
 *        If the final calculation result is beyond the range of int16,
 *        the calculation result is not guaranteed.
 * @tparam T: Input and output data types, int16
 * @tparam isReuseSrc: Whether temporary variables can reuse the input memory.
 * @param [out] dstTensor: output LocalTensor, the minimum shape is 16.
 * @param [in] src0Tensor: input0 LocalTensor
 * @param [in] src1Tensor: input1 LocalTensor
 * @param [in] calCount: amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void ReduceXorSum(LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
                                    const LocalTensor<T>&src1Tensor, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(std::is_same<T, int16_t>::value, "ReduceXorSum only support int16_t data type on current device!");

    LocalTensor<uint8_t> tmp;
    const bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(tmp);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    ReduceXorSumCompute<T, isReuseSource>(dstTensor, src0Tensor, src1Tensor, tmp, calCount);
}
#pragma end_pipe
}  // namespace AscendC

#endif

#endif  // LIB_REDUCE_XOR_SUM_REDUCE_XOR_SUM_H
