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
 * \file ascend_quant_v220_impl.h
 * \brief
 */
#ifndef IMPL_QUANTIZATION_QUANT_ASCEND_QUANT_V220_IMPL_H
#define IMPL_QUANTIZATION_QUANT_ASCEND_QUANT_V220_IMPL_H
#include "ascend_quant_pre_impl.h"

namespace AscendC {
// api impl
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuantImpl(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendQuantCalc<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scale, offset, calCount);
}

template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuantImpl(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const T offset, const uint32_t scaleCount, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendQuantCalc<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor,
        offset, scaleCount, calCount);
}

template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuantImpl(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const LocalTensor<T>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount,
    const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendQuantCalc<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor,
        offsetTensor, scaleCount, offsetCount, calCount);
}

} //  namespace AscendC
#endif // IMPL_QUANTIZATION_QUANT_ASCEND_QUANT_V220_IMPL_H
