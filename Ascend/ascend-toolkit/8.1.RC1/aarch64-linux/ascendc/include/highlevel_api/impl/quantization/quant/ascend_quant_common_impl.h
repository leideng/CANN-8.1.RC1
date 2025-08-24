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
 * \file ascend_quant_common_impl.h
 * \brief
 */
#ifndef IMPL_QUANTIZATION_QUANT_ASCEND_QUANT_COMMON_IMPL_H
#define IMPL_QUANTIZATION_QUANT_ASCEND_QUANT_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"

#if __CCE_AICORE__ == 220
#include "ascend_quant_v220_impl.h"
#elif __CCE_AICORE__ == 200
#include "ascend_quant_v200_impl.h"
#elif __CCE_AICORE__ == 100
#include "ascend_quant_v100_impl.h"
#endif

namespace AscendC {
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuantImpl(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<T>& srcTensor, const float scale, const float offset, uint32_t calCount)
{
    LocalTensor<uint8_t> stackTensor;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackTensor);
    ASCENDC_ASSERT((ans),
                   { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, stackTensor, scale, offset, calCount);
}

template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuantImpl(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& scaleTensor, const T offset, const uint32_t scaleCount, const uint32_t calCount)
{
    LocalTensor<uint8_t> stackTensor;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackTensor);
    ASCENDC_ASSERT((ans),
                   { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, stackTensor, scaleTensor, offset, scaleCount,
        calCount);
}

template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuantImpl(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor, const uint32_t scaleCount,
    const uint32_t offsetCount, const uint32_t calCount)
{
    LocalTensor<uint8_t> stackTensor;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackTensor);
    ASCENDC_ASSERT((ans),
                   { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, stackTensor, scaleTensor, offsetTensor,
        scaleCount, offsetCount, calCount);
}

template <typename T>
__aicore__ inline void IsQuantParamValid(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const LocalTensor<T>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount,
    const uint32_t calCount)
{
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should not larger than srcTensor size %u.",
            calCount, srcTensor.GetSize());
    });
    ASCENDC_ASSERT((scaleCount <= scaleTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "scaleCount is %u, which should not larger than scaleTensor size %u.",
            scaleCount, scaleTensor.GetSize());
    });
    ASCENDC_ASSERT((offsetCount <= offsetTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "offsetCount is %u, which should not larger than offsetTensor size %u.",
            offsetCount, offsetTensor.GetSize());
    });
    ASCENDC_ASSERT((scaleCount == offsetCount && scaleCount > 0), {
        KERNEL_LOG(KERNEL_ERROR, "scaleCount is %u, which should be equal to offsetCount %u and not zero.",
            scaleCount, offsetCount);
    });
    ASCENDC_ASSERT((calCount % scaleCount == 0 && calCount > 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be integral multiple of scaleCount %u and not zero.",
            calCount, scaleCount);
    });
}
template <typename T>
__aicore__ inline void IsQuantParamValid(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const T& offset, const uint32_t scaleCount, const uint32_t calCount)
{
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should not larger than srcTensor size %u.",
            calCount, srcTensor.GetSize());
    });
    ASCENDC_ASSERT((scaleCount <= scaleTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "scaleCount is %u, which should not larger than scaleTensor size %u.",
            scaleCount, scaleTensor.GetSize());
    });
    ASCENDC_ASSERT((scaleCount > 0), {
        KERNEL_LOG(KERNEL_ERROR, "scaleCount is %u, which should be not zero.",
            scaleCount);
    });
    ASCENDC_ASSERT((calCount % scaleCount == 0 && calCount > 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be integral multiple of scaleCount %u and not zero.",
            calCount, scaleCount);
    });
}

}  // namespace AscendC
#endif  // IMPL_QUANTIZATION_QUANT_ASCEND_QUANT_COMMON_IMPL_H
