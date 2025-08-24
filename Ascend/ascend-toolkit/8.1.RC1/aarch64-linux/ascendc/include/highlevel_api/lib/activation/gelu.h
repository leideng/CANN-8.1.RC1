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
 * \file gelu.h
 * \brief
 */
#ifndef LIB_GELU_GELU_H
#define LIB_GELU_GELU_H
#include "kernel_tensor.h"
#include "../../impl/activation/gelu/gelu_impl.h"
namespace AscendC {
#pragma begin_pipe(V)
/* !
 * \brief Gelu(x) = x / (1 + e ^ (-1.59576912 * (x + 0.044715 * x ^ 3)))
 * \param [out] dstLocal, output LocalTensor
 * \param [in] srcLocal, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] dataSize, number of input data elements
 * \param [in] highPrecision, whether to enable the high-precision interface to improve the calculation accuracy
 * \param [in] highPerformance, whether to enable the high-performance interface to improve the computing efficiency
 */
template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void Gelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize)
{
    if ASCEND_IS_AIC {
        return;
    }
    GeluImpl<T, highPrecision, highPerformance>(dstLocal, srcLocal, sharedTmpBuffer, dataSize);
}

/* !
 * \ingroup Gelu
 * \param [out] dstLocal, output LocalTensor
 * \param [in] srcLocal, input LocalTensor
 * \param [in] dataSize, number of input data elements
 * \param [in] highPrecision, whether to enable the high-precision interface to improve the calculation accuracy
 * \param [in] highPerformance, whether to enable the high-performance interface to improve the computing efficiency
 */
template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void Gelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint32_t dataSize)
{
    if ASCEND_IS_AIC {
        return;
    }
    GeluImpl<T, highPrecision, highPerformance>(dstLocal, srcLocal, dataSize);
}

/* !
 * \brief FasterGelu(x) = x / (1 + e ^ (-1.702 * x))
 * \param [out] dstLocal, output LocalTensor
 * \param [in] srcLocal, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] dataSize, number of input data elements
 * \param [in] highPrecision, whether to enable the high-precision interface to improve the calculation accuracy
 * \param [in] highPerformance, whether to enable the high-performance interface to improve the computing efficiency
 */
template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void FasterGelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize)
{
    if ASCEND_IS_AIC {
        return;
    }
    FasterGeluImpl<T, highPrecision, highPerformance>(dstLocal, srcLocal, sharedTmpBuffer, dataSize);
}

/* !
 * \ingroup FasterGelu
 * \param [out] dstLocal, output LocalTensor
 * \param [in] srcLocal, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] dataSize, number of input data elements
 * \param [in] highPrecision, whether to enable the high-precision interface to improve the calculation accuracy
 * \param [in] highPerformance, whether to enable the high-performance interface to improve the computing efficiency
 */
template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void FasterGelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint32_t dataSize)
{
    if ASCEND_IS_AIC {
        return;
    }
    FasterGeluImpl<T, highPrecision, highPerformance>(dstLocal, srcLocal, dataSize);
}

/* !
 * \brief sgn(x) = (x + 0.000000000001) / |(x + 0.000000000001)|
 * \brief fast_gelu_v2(x) = x * (sgn(x) * [(-0.1444) * (clip(|0.7071 * x|, max=1.769) - 1.769) ^ 2 + 0.5] + 0.5)
 * \param [out] dstLocal, output LocalTensor
 * \param [in] srcLocal, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] dataSize, number of input data elements
 * \param [in] highPrecision, whether to enable the high-precision interface to improve the calculation accuracy
 * \param [in] highPerformance, whether to enable the high-performance interface to improve the computing efficiency
 */
template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void FasterGeluV2(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize)
{
    if ASCEND_IS_AIC {
        return;
    }
    ASCENDC_ASSERT((IsSameType<T, half>::value || IsSameType<T, float>::value),
        { KERNEL_LOG(KERNEL_ERROR, "FasterGeluV2 only support data type: float/half"); });
    FasterGeluV2Impl<T, highPrecision, highPerformance>(dstLocal, srcLocal, sharedTmpBuffer, dataSize);
}

/* !
 * \ingroup FasterGeluV2
 * \param [out] dstLocal, output LocalTensor
 * \param [in] srcLocal, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] dataSize, number of input data elements
 * \param [in] highPrecision, whether to enable the high-precision interface to improve the calculation accuracy
 * \param [in] highPerformance, whether to enable the high-performance interface to improve the computing efficiency
 */
template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void FasterGeluV2(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint32_t dataSize)
{
    if ASCEND_IS_AIC {
        return;
    }
    ASCENDC_ASSERT((IsSameType<T, half>::value || IsSameType<T, float>::value),
        { KERNEL_LOG(KERNEL_ERROR, "FasterGeluV2 only support data type: float/half"); });
    FasterGeluV2Impl<T, highPrecision, highPerformance>(dstLocal, srcLocal, dataSize);
}
#pragma end_pipe
} // namespace AscendC
#endif // LIB_GELU_GELU_H
