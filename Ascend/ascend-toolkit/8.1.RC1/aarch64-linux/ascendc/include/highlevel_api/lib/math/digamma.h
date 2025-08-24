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
 * \file digamma.h
 * \brief
 */
#ifndef LIB_MATH_DIGAMMA_H
#define LIB_MATH_DIGAMMA_H

#if __CCE_AICORE__ >= 200

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../impl/math/digamma/digamma_common_impl.h"

namespace AscendC {
#pragma begin_pipe(V)
/*
 * @ingroup Digamma
 * @brief Computes the logarithmic derivative of the gamma function on input. f(x) = digamma(x)
 * @tparam T: Input and output data types, half or float.
 * @tparam isReuseSrc: Whether temporary variables can reuse the input memory.
 * @param [out] dstTensor: output LocalTensor
 * @param [in] srcTensor: input LocalTensor
 * @param [in] sharedTmpBuffer: input local temporary Tensor
 * @param [in] calCount: amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Digamma(LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                               LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    DigammaCompute<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

/*
 * @ingroup Digamma
 * @brief Computes the logarithmic derivative of the gamma function on input. f(x) = digamma(x)
 * @tparam T: Input and output data types, half or float.
 * @tparam isReuseSrc: Whether temporary variables can reuse the input memory.
 * @param [out] dstTensor: output LocalTensor
 * @param [in] srcTensor: input LocalTensor
 * @param [in] calCount: amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Digamma(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    LocalTensor<uint8_t> tmp;
    const bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(tmp);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    DigammaCompute<T, isReuseSource>(dstTensor, srcTensor, tmp, calCount);
}
#pragma end_pipe
} // namespace AscendC

#endif // __CCE_AICORE__ >= 200

#endif // LIB_MATH_DIGAMMA_H
