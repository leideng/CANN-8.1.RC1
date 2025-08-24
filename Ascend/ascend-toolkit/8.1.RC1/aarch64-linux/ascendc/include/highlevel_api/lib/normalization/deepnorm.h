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
 * \file deepnorm.h
 * \brief Use deepnorm for the normalization process.   DeepNorm(x) = LayerNorm(alpha * x + Sublayer(x))
 *        LayerNorm(x) = gamma * (x - mean) / (variance ^ 2 + epsilon) + beta
 *        For more info of DeepNorm, please check https://arxiv.org/abs/2203.00555
 */
#ifndef LIB_NORMALIZATION_DEEPNORM_H
#define LIB_NORMALIZATION_DEEPNORM_H
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
#include "kernel_tensor.h"
#include "../../impl/normalization/deepnorm/deepnorm_common_impl.h"

namespace AscendC {
#pragma begin_pipe(V)

/*!
 * \ingroup DeepNorm
 * \note User should make sure that memory on UB is big enough for input tiling.
 * \tparam T: Data type to be calculated, half or float
 * \tparam isReuseSrc: Whether to reuse the buffer of srcTensor.
 *                     If the value is true, srcTensor can used as tmpBuffer and the data in it will be overwritten.
 *                     If the value is false, srcTensor will not be used as tmpBuffer for calculation.
 * \tparam isBasicBlock: Whether the srcTensor shape [B, S, H] meets the following requirement
 *                     1. H % 64 = 0   2. H <= 2040   3. B * S % 8 = 0
 *                     If shape meets the rules above, isBasicBlock can be set true and the performance will be better.
 *                     Otherwise, the performance remains the same.
 * \param [out] dstLocal: output localTensor. Shape is [B, S, H]
 * \param [out] meanLocal: output localTensor for mean result. Shape is [B, S]
 * \param [out] rstdLocal: output localTensor for variance result.Shape is [B, S]
 * \param [in] srcLocal: Input localTensor x for calculation. Shape is [B, S, H]
 * \param [in] gxLocal: Input localTensor Sublayer(x) for calculation. Shape is [B, S, H]
 * \param [in] betaLocal: Input localTensor beta used in layernorm. Shape is [H].
 * \param [in] gammaLocal: Input localTensor gamma used in layernorm. Shape is [H].
 * \param [in] sharedTmpBuffer: extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             deepnorm_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] alpha: alpha param in DeepNorm formula
 * \param [in] epsilon: epsilon param in LayerNorm formula
 * \param [in] tiling: Tiling information required for DeepNorm calculation.
 */
template <typename T, bool isReuseSrc = false, bool isBasicBlock = false>
__aicore__ inline void DeepNorm(const LocalTensor<T>& dstLocal, const LocalTensor<T>& meanLocal,
    const LocalTensor<T>& rstdLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& gxLocal,
    const LocalTensor<T>& betaLocal, const LocalTensor<T>& gammaLocal, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const T alpha, const T epsilon, DeepNormTiling& tiling)
{
    if ASCEND_IS_AIC {
        return;
    }
    DeepNormAPI::DeepNormImpl<T, isReuseSrc, isBasicBlock>(dstLocal, meanLocal, rstdLocal, srcLocal, gxLocal, betaLocal,
        gammaLocal, sharedTmpBuffer, alpha, epsilon, tiling);
}

/*!
 * \ingroup DeepNorm
 * \note User should make sure that memory on UB is big enough for input tiling.
 * \tparam T: Data type to be calculated, half or float
 * \tparam isReuseSrc: Whether to reuse the buffer of srcTensor.
 *                     If the value is true, srcTensor can used as tmpBuffer and the data in it will be overwritten.
 *                     If the value is false, srcTensor will not be used as tmpBuffer for calculation.
 * \tparam isBasicBlock: Whether the srcTensor shape [B, S, H] meets the following requirement
 *                     1. H % 64 = 0   2. H <= 2040   3. B * S % 8 = 0
 *                     If shape meets the rules above, isBasicBlock can be set true and the performance will be better.
 *                     Otherwise, the performance remains the same.
 * \param [out] dstLocal: output localTensor. Shape is [B, S, H]
 * \param [out] meanLocal: output localTensor for mean result. Shape is [B, S]
 * \param [out] rstdLocal: output localTensor for variance result.Shape is [B, S]
 * \param [in] srcLocal: Input localTensor x for calculation. Shape is [B, S, H]
 * \param [in] gxLocal: Input localTensor Sublayer(x) for calculation. Shape is [B, S, H]
 * \param [in] betaLocal: Input localTensor beta used in layernorm. Shape is [H].
 * \param [in] gammaLocal: Input localTensor gamma used in layernorm. Shape is [H].
 * \param [in] alpha: alpha param in DeepNorm formula
 * \param [in] epsilon: epsilon param in LayerNorm formula
 * \param [in] tiling: Tiling information required for DeepNorm calculation.
 */
template <typename T, bool isReuseSrc = false, bool isBasicBlock = false>
__aicore__ inline void DeepNorm(const LocalTensor<T>& dstLocal, const LocalTensor<T>& meanLocal,
    const LocalTensor<T>& rstdLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& gxLocal,
    const LocalTensor<T>& betaLocal, const LocalTensor<T>& gammaLocal, const T alpha, const T epsilon,
    DeepNormTiling& tiling)
{
    if ASCEND_IS_AIC {
        return;
    }
    DeepNormAPI::DeepNormImpl<T, isReuseSrc, isBasicBlock>(dstLocal, meanLocal, rstdLocal, srcLocal, gxLocal, betaLocal,
        gammaLocal, alpha, epsilon, tiling);
}
#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_NORMALIZATION_DEEPNORM_H
