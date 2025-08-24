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
 * \file ceil.h
 * \brief
 */

#ifndef LIB_MATH_CEIL_H
#define LIB_MATH_CEIL_H

#include "kernel_tensor.h"
#include "../../impl/math/ceil/ceil_common_impl.h"

#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220

namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \brief the ceil function maps x to the smallest integer greater than or equal to x
 * (e.g. ceil(2.4) is 3, ceil(-2.4) is -2).
 * For details about the interface description, see
 * https://pytorch.org/docs/stable/generated/torch.ceil.html.
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Ceil(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount)
{
#if __CCE_AICORE__ == 220
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
#endif
    CeilImpl(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

/*!
 * \ingroup Ceil
 * \note support data type: half and float
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Ceil(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const uint32_t calCount)
{
#if __CCE_AICORE__ == 220
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
#endif
    CeilImpl(dstTensor, srcTensor, calCount);
}

#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_MATH_CEIL_H