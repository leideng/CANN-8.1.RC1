/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layernormgrad.h
 * \brief
 */
#ifndef LIB_NORMALIZATION_LAYERNORMGRAD_H
#define LIB_NORMALIZATION_LAYERNORMGRAD_H

#include "kernel_tensor.h"
#include "../../impl/normalization/layernormgrad/layernormgrad_common_impl.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \brief layernormgrad intf function:
 * tips : x ^ y = exp (y * ln(x))
 * main process:
 * x1Tensor = inputDy * inputGamma
 * x2Tensor = inputX - inputMean
 * pd_var = np.sum(((-0.5) * x1Tensor * (x2Tensor) * np.power((inputVariace + EPSLON), (-1.5))), reduce_axis,
 * keepdims=True)
 * pd_mean = np.sum(((-1.0) * x1Tensor * np.power((inputVariace + EPSLON), (-0.5))), reduce_axis, keepdims = True) +
 * pd_var * (1.0/m) * np.sum(((-2.0) * (x2Tensor)), reduce_axis, keepdims=True)
 * pd_x = x1Tensor * np.power((inputVariace + EPSLON), (-0.5)) +
 * pd_var*(2.0 / H)*(x2Tensor) +
 * pd_mean*(1.0 / H)
 *
 * \note support data type: half and float
 *
 * \param [out] outputPdX, output LocalTensor, shape is [B, S, H]
 * \param [out] resForGamma, output LocalTensor, shape is [B, S, H]
 * \param [in] inputDy, input LocalTensor, shape is [B, S, H]
 * \param [in] inputX, input LocalTensor, shape is [B, S, H]
 * \param [in] inputVariance, input LocalTensor, shape is [B, S, 1]
 * \param [in] inputMean, input LocalTensor, shape is [B, S, 1]
 * \param [in] inputGamma, input LocalTensor, shape is [H]
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] epsilon, weighting factor
 * \param [in] tiling, layernormgrad tiling
 * \param [in] shapeInfo, LayerNormGrad Shape Info
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void LayerNormGrad(const LocalTensor<T> &outputPdX, const LocalTensor<T> &resForGamma,
    const LocalTensor<T> &inputDy, const LocalTensor<T> &inputX, const LocalTensor<T> &inputVariance,
    const LocalTensor<T> &inputMean, const LocalTensor<T> &inputGamma, LocalTensor<uint8_t> &sharedTmpBuffer, T epsilon,
    LayerNormGradTiling &tiling, const LayerNormGradShapeInfo &shapeInfo = {})
{
    LayerNormGradImpl<T, isReuseSource>(outputPdX, resForGamma, inputDy, inputX, inputVariance, inputMean, inputGamma,
        sharedTmpBuffer, epsilon, tiling, shapeInfo);
}
/* !
 * \brief layernormgrad intf function
 *
 * \note support data type: half and float
 *
 * \param [out] outputPdX, output LocalTensor, shape is [B, S, H]
 * \param [out] resForGamma, output LocalTensor, shape is [B, S, H]
 * \param [in] inputDy, input LocalTensor, shape is [B, S, H]
 * \param [in] inputX, input LocalTensor, shape is [B, S, H]
 * \param [in] inputVariance, input LocalTensor, shape is [B, S, 1]
 * \param [in] inputMean, input LocalTensor, shape is [B, S, 1]
 * \param [in] inputGamma, input LocalTensor, shape is [H]
 * \param [in] epsilon, weighting factor
 * \param [in] tiling, layernormgrad tiling
 * \param [in] shapeInfo, LayerNormGrad Shape Info
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void LayerNormGrad(const LocalTensor<T> &outputPdX, const LocalTensor<T> &resForGamma,
    const LocalTensor<T> &inputDy, const LocalTensor<T> &inputX, const LocalTensor<T> &inputVariance,
    const LocalTensor<T> &inputMean, const LocalTensor<T> &inputGamma, T epsilon, LayerNormGradTiling &tiling,
    const LayerNormGradShapeInfo &shapeInfo = {})
{
    LayerNormGradImpl<T, isReuseSource>(outputPdX, resForGamma, inputDy, inputX, inputVariance, inputMean, inputGamma,
        epsilon, tiling, shapeInfo);
}
#pragma end_pipe
} // namespace AscendC
#endif // LIB_NORMALIZATION_LAYERNORMGRAD_H
