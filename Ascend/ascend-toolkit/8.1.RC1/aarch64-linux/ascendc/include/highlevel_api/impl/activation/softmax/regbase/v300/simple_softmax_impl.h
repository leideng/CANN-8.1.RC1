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
 * \file simple_softmax_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_V300_SIMPLE_SOFTMAX_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_V300_SIMPLE_SOFTMAX_IMPL_H

#include "softmax_common_impl.h"

namespace AscendC {
template <typename T>
__aicore__ inline void SoftMaxLogV2NZImpl(const LocalTensor<T>& dst, const LocalTensor<float>& inLogSumExpTensor,
    const LocalTensor<T>& src, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, {
        KERNEL_LOG(KERNEL_ERROR, "softmaxlogv2 format NZ is not supported on current device!");
    });
}
template <typename T, bool isBasicBlock = false>
__aicore__ inline void SoftMaxLogSumExpImpl(const LocalTensor<T>& dst, const LocalTensor<float>& inLogSumExpTensor,
    const LocalTensor<T>& src, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, {
        KERNEL_LOG(KERNEL_ERROR, "softmaxlogv2 is not supported on current device!");
    });
}
template <typename T1, typename T2>
__aicore__ inline void SimpleSoftMaxNZImpl(const LocalTensor<T1>& dst, const LocalTensor<T1>& inSumTensor,
    const LocalTensor<T1>& inMaxTensor, const LocalTensor<T1>& src, const LocalTensor<float> workLocal,
    const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, {
        KERNEL_LOG(KERNEL_ERROR, "simplesoftmax format NZ is not supported on current device!");
    });
}
template <typename T1, typename T2>
__aicore__ inline void SimpleSoftMaxNZImpl(const LocalTensor<half>& dst, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, {
        KERNEL_LOG(KERNEL_ERROR, "simplesoftmax format NZ is not supported on current device!");
    });
}

__aicore__ inline void SimpleSoftMaxGenericNDImpl(const LocalTensor<half>& dst, const LocalTensor<half>& inSumTensor,
    const LocalTensor<half>& inMaxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const uint32_t offset1, const uint32_t offset2, const uint32_t curSplitM)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitSize];
    const uint32_t splitSize = curSplitM * tiling.splitK;
    const uint32_t reduceSize = curSplitM * tiling.reduceK;

    Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
    Cast(tmpBuffer2, inMaxTensor[offset2], RoundMode::CAST_NONE, reduceSize);
    SubNDImpl(tmpBuffer0, tmpBuffer0, tmpBuffer2, curSplitM, tiling.srcK, tiling.reduceK);
    Exp(tmpBuffer0, tmpBuffer0, splitSize);
    Cast(tmpBuffer2, inSumTensor[offset2], RoundMode::CAST_NONE, reduceSize);
    DivNDImpl(tmpBuffer0, tmpBuffer0, tmpBuffer2, curSplitM, tiling.srcK, tiling.reduceK);
    Cast(dst[offset1], tmpBuffer0, RoundMode::CAST_ROUND, splitSize);
}
__aicore__ inline void SimpleSoftMaxGenericNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const uint32_t offset1, const uint32_t offset2, const uint32_t curSplitM)
{
    const uint32_t splitSize = curSplitM * tiling.splitK;
    SubNDImpl(dst[offset1], src[offset1], inMaxTensor[offset2], curSplitM, tiling.srcK, tiling.reduceK);
    Exp(dst[offset1], dst[offset1], splitSize);
    DivNDImpl(dst[offset1], dst[offset1], inSumTensor[offset2], curSplitM, tiling.srcK, tiling.reduceK);
}

template <typename T, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxNDImpl(const LocalTensor<T>& dst, const LocalTensor<T>& inSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<T>& src, const LocalTensor<float> workLocal,
    const SoftMaxTiling& tiling)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        SimpleSoftMaxGenericNDImpl(dst, inSumTensor, inMaxTensor, src, workLocal, tiling, 0, 0, tiling.srcM);
    } else {
        uint32_t offset1 = 0;
        uint32_t offset2 = 0;
        uint32_t curSplitM = tiling.splitM;
        for (uint32_t i = 0; i <= tiling.rangeM; i++) {
            offset1 = i * tiling.splitSize;
            offset2 = i * tiling.reduceSize;
            SimpleSoftMaxGenericNDImpl(dst, inSumTensor, inMaxTensor, src, workLocal, tiling, offset1, offset2,
                curSplitM);
            if (i == (tiling.rangeM - 1)) {
                if (tiling.tailM == 0) {
                    break;
                }
                offset1 = tiling.rangeM * tiling.splitSize;
                offset2 = tiling.rangeM * tiling.reduceSize;
                curSplitM = tiling.tailM;
            }
        }
    }
}

template <typename T, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxNDImpl(const LocalTensor<half>& dst, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "simplesoftmax current data type is not supported on current device!"); });
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_V300_SIMPLE_SOFTMAX_IMPL_H