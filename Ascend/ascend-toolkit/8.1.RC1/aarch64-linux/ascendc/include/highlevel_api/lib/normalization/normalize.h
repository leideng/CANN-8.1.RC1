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
 * \file noramlize.h
 * \brief Given mean and variance, calculate rstd and output.
 */
#ifndef LIB_NORMALIZATION_NORMALIZE_H
#define LIB_NORMALIZATION_NORMALIZE_H
#include "lib/normalization/normalize_utils.h"
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
#include "kernel_tensor.h"
#include "../../impl/normalization/normalize/normalize_common_impl.h"

namespace AscendC {
#pragma begin_pipe(V)

/*!
 * \brief Applies Layer Normalization over a mini-batch of inputs, 
 *        performs the operation of taking the reciprocal of the standard deviation of the intermediate output results.
 *
 * \note support data type: half and float
 *
 * \param [out] output, output LocalTensor, shape is [A, R]
 * \param [out] outputRstd, output LocalTensor, shape is [A]
 * \param [in] inputMean, input LocalTensor, shape is [A]
 * \param [in] inputVariance, input LocalTensor, shape is [A]
 * \param [in] inputX, input LocalTensor, shape is [A, R]
 * \param [in] gamma, input LocalTensor, shape is [R]
 * \param [in] beta, input LocalTensor, shape is [R]
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] epsilon, weighting factor, prevent division by zero.
 * \param [in] para, NormalizePara struct, contains aLength, rLength and rLengthWithPadding.
 */
template <typename U, typename T, bool isReuseSource = false, const NormalizeConfig& config = NLCFG_NORM>
__aicore__ inline void Normalize(const LocalTensor<T>& output, const LocalTensor<float>& outputRstd,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<T>& inputX,
    const LocalTensor<U>& gamma, const LocalTensor<U>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const float epsilon, const NormalizePara& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    NormalizeImpl<U, T, isReuseSource, config>(output, outputRstd, inputMean, inputVariance, inputX, gamma, beta,
        sharedTmpBuffer, epsilon, para);
}

/*!
 * \brief Applies Layer Normalization over a mini-batch of inputs, 
 *        performs the operation of taking the reciprocal of the standard deviation of the intermediate output results.
 *
 * \note support data type: half and float
 *
 * \param [out] output, output LocalTensor, shape is [A, R]
 * \param [out] outputRstd, output LocalTensor, shape is [A]
 * \param [in] inputMean, input LocalTensor, shape is [A]
 * \param [in] inputVariance, input LocalTensor, shape is [A]
 * \param [in] inputX, input LocalTensor, shape is [A, R]
 * \param [in] gamma, input LocalTensor, shape is [R]
 * \param [in] beta, input LocalTensor, shape is [R]
 * \param [in] epsilon, weighting factor, prevent division by zero.
 * \param [in] para, NormalizePara struct, contains aLength, rLength and rLengthWithPadding.
 */
template <typename U, typename T, bool isReuseSource = false, const NormalizeConfig& config = NLCFG_NORM>
__aicore__ inline void Normalize(const LocalTensor<T>& output, const LocalTensor<float>& outputRstd,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<T>& inputX,
    const LocalTensor<U>& gamma, const LocalTensor<U>& beta, const float epsilon, const NormalizePara& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    NormalizeImpl<U, T, isReuseSource, config>(output, outputRstd, inputMean, inputVariance, inputX, gamma, beta,
        epsilon, para);
}
#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_NORMALIZATION_NORMALIZE_H
