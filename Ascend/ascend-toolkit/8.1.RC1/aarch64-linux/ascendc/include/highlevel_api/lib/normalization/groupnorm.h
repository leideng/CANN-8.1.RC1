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
 * \file groupnorm.h
 * \brief
 */

#ifndef LIB_NORMALIZATION_GROUPNORM_H
#define LIB_NORMALIZATION_GROUPNORM_H
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220

#include "kernel_tensor.h"
#include "../../impl/normalization/groupnorm/groupnorm_common_impl.h"
#include "kernel_tiling/kernel_tiling.h"
namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \brief Applies Group Normalization over a mini-batch of inputs as described in the paper Group Normalization.
 *
 * \note support data type: half and float
 *
 * \param [out] output, output LocalTensor, shape is [n, C, H, W]
 * \param [out] outputMean, output LocalTensor, shape is [n, groupNum]
 * \param [out] outputVariance, output LocalTensor, shape is [n, groupNum]
 * \param [in] inputX, input LocalTensor, shape is [n, C, H, W]
 * \param [in] gamma, input LocalTensor, shape is [C]
 * \param [in] beta, input LocalTensor, shape is [C]
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] epsilon, weighting factor
 * \param [in] tiling, groupnormtiling
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void GroupNorm(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
    const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamma,
    const LocalTensor<T>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon, GroupNormTiling& tiling)
{
    GroupNormImpl<T, isReuseSource>(output, outputMean, outputVariance, inputX, gamma, beta, sharedTmpBuffer, epsilon,
        tiling);
}

/*!
 * \brief Applies Group Normalization over a mini-batch of inputs as described in the paper Group Normalization.
 *
 * \note support data type: half and float
 *
 * \param [out] output, output LocalTensor, shape is [n, C, H, W]
 * \param [out] outputMean, output LocalTensor, shape is [n, groupNum]
 * \param [out] outputVariance, output LocalTensor, shape is [n, groupNum]
 * \param [in] inputX, input LocalTensor, shape is [n, C, H, W]
 * \param [in] gamma, input LocalTensor, shape is [C]
 * \param [in] beta, input LocalTensor, shape is [C]
 * \param [in] epsilon, weighting factor
 * \param [in] tiling, groupnormtiling
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void GroupNorm(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
    const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamma,
    const LocalTensor<T>& beta, const T epsilon, GroupNormTiling& tiling)
{
    GroupNormImpl<T, isReuseSource>(output, outputMean, outputVariance, inputX, gamma, beta, epsilon, tiling);
}
#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_NORMALIZATION_GROUPNORM_H