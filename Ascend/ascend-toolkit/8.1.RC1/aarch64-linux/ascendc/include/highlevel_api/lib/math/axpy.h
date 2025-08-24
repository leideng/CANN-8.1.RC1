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
 * \file axpy.h
 * \brief Compute axpy   dst = src * scalar + dst
 */
#ifndef LIB_MATH_AXPY_H
#define LIB_MATH_AXPY_H
#include "kernel_tensor.h"
#include "../../impl/math/axpy/axpy_common_impl.h"

namespace AscendC {
#pragma begin_pipe(V)

/*!
 * \ingroup Axpy
 * \brief compute axpy   dst = src * scalar + dst
 * \tparam isReuseSource: Whether to reuse the buffer of srcTensor.
 *         If the value is true, srcTensor can used as tmpBuffer and the data in it will be overwritten.
 *         If the value is false, srcTensor will not be used as tmpBuffer for calculation.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] scalarValue: input scalarValue
 * \param [in] sharedTmpBuffer: extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at axpy_tiling.h.
 *             Generally, the more space you allocate, the better performance you will achieve, and the performance
 *             reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it is not guaranteed
 *             that the shared space will be cleared after usage, the data could be anything.
 * \param [in] calCount: input total calCount
 */
template <typename T, typename U, bool isReuseSource = false>
__aicore__ inline void Axpy(const LocalTensor<T>& dstTensor, const LocalTensor<U>& srcTensor, const U scalarValue,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    AxpyImpl<T, U, isReuseSource>(dstTensor, srcTensor, scalarValue, sharedTmpBuffer, calCount);
}

#pragma end_pipe
} // namespace AscendC
#endif // LIB_MATH_AXPY_H