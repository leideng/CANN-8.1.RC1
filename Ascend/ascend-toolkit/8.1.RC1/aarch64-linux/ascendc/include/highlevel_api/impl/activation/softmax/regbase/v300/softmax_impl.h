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
 * \file softmax_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_V300_SOFTMAX_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_V300_SOFTMAX_IMPL_H

#include "softmax_common_impl.h"

namespace AscendC {
template <typename T1, typename T2, bool isBasicBlock = false>
__aicore__ inline void SoftMaxNZImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "softmax format NZ is not supported on current device!"); });
}
template <bool isFlashV2 = false>
__aicore__ inline void SoftMaxGenericNDImpl(const LocalTensor<half>& dst, const LocalTensor<half>& sumTensor,
    const LocalTensor<half>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const uint32_t& offset1, const uint32_t& offset2, const uint32_t& splitSize,
    const uint32_t& reduceSize, const ReduceLastND& reduceParam)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal[0];
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitSize];
    const LocalTensor<float>& tmpBuffer3 = workLocal[tiling.splitSize + tiling.reduceSize]; // need splitM * 64

    auto halfBuffer0 = tmpBuffer0.ReinterpretCast<half>();
    ReduceMaxImpl(maxTensor[offset2], src[offset1], halfBuffer0, reduceParam);
    Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
    Cast(tmpBuffer2, maxTensor[offset2], RoundMode::CAST_NONE, reduceSize);

    SubNDImpl(tmpBuffer0, tmpBuffer0, tmpBuffer2, reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);

    Exp(tmpBuffer0, tmpBuffer0, splitSize);
    ReduceSumImpl(tmpBuffer2, tmpBuffer0, tmpBuffer3, reduceParam);

    Cast(sumTensor[offset2], tmpBuffer2, RoundMode::CAST_ROUND, reduceSize);
    if constexpr (!isFlashV2) {
        DivNDImpl(tmpBuffer0, tmpBuffer0, tmpBuffer2, reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);
    }
    Cast(dst[offset1], tmpBuffer0, RoundMode::CAST_ROUND, splitSize);
}

template <bool isFlashV2 = false>
__aicore__ inline void SoftMaxGenericNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<float>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const uint32_t& offset1, const uint32_t& offset2, const uint32_t& splitSize,
    const uint32_t& reduceSize, const ReduceLastND& reduceParam)
{
    const LocalTensor<float>& tmpBuffer1 = workLocal[0]; // need splitM * 64

    ReduceMaxImpl(maxTensor[offset2], src[offset1], tmpBuffer1, reduceParam);
    SubNDImpl(dst[offset1], src[offset1], maxTensor[offset2], reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);

    Exp(dst[offset1], dst[offset1], splitSize);

    ReduceSumImpl(sumTensor[offset2], dst[offset1], tmpBuffer1, reduceParam);
    if constexpr (!isFlashV2) {
        DivNDImpl(dst[offset1], dst[offset1], sumTensor[offset2], reduceParam.originalSrcM, tiling.srcK,
            tiling.reduceK);
    }
}

template <typename T1, typename T2, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxNDImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    ReduceLastND reduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
        tiling.splitK, tiling.reduceM,     tiling.reduceK };
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitSize = tiling.splitSize;
    uint32_t reduceSize = tiling.reduceSize;

    for (uint32_t i = 0; i <= tiling.rangeM; i++) {
        SoftMaxGenericNDImpl(dst, sumTensor, maxTensor, src, workLocal, tiling, offset1, offset2, splitSize, reduceSize,
            reduceParam);
        offset1 += tiling.splitSize;
        offset2 += tiling.reduceSize;
        if (i == (tiling.rangeM - 1)) {
            if (tiling.tailM == 0) {
                break;
            }
            offset2 = tiling.rangeM * tiling.reduceSize;
            offset1 = tiling.rangeM * tiling.splitSize;
            splitSize = tiling.tailSplitSize;
            reduceSize = tiling.tailReduceSize;
            reduceParam.originalSrcM = tiling.tailM;
            reduceParam.srcM = tiling.tailM;
            reduceParam.dstM = tiling.tailM;
        }
    }
}
template <typename T1, typename T2, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxNDImpl(const LocalTensor<half>& dst, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "softmax current data type is not supported on current device!"); });
}
template <typename T1, typename T2, bool isBasicBlock = false>
__aicore__ inline void SoftMaxNZImpl(const LocalTensor<half>& dst, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "softmax current data type is not supported on current device!"); });
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_V300_SOFTMAX_IMPL_H