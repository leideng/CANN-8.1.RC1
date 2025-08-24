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
 * \file softmax_flashv2_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_V300_SOFTMAX_FLASHV2_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_V300_SOFTMAX_FLASHV2_IMPL_H

#include "softmax_common_impl.h"

namespace AscendC {
template <typename T1, typename T2, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV2Update(const LocalTensor<half>& dst, const LocalTensor<float>& expSumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "softmaxflashv2 is not supported on current device!"); });
}
template <typename T1, typename T2, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV2NoUpdate(const LocalTensor<T1>& dst, const LocalTensor<T1>& expSumTensor,
    const LocalTensor<T1>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitSize = tiling.splitSize;
    uint32_t reduceSize = tiling.reduceSize;
    ReduceLastND reduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
        tiling.splitK, tiling.reduceM,     tiling.reduceK };
    for (uint32_t i = 0; i <= tiling.rangeM; i++) {
        SoftMaxGenericNDImpl<true>(dst, expSumTensor, maxTensor, src, workLocal, tiling, offset1, offset2, splitSize,
            reduceSize, reduceParam);
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

__aicore__ inline void SoftmaxFlashV2UpdateImpl(const LocalTensor<half>& dst, const LocalTensor<half>& sumTensor,
    const LocalTensor<half>& maxTensor, const LocalTensor<half>& src, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<half>& inSumTensor, const LocalTensor<half>& inMaxTensor, const LocalTensor<float>& workLocal,
    const ReduceLastND& reduceParam, const SoftMaxTiling& tiling, const uint32_t& offset1, const uint32_t& offset2,
    const uint32_t& splitSize, const uint32_t& reduceSize)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;                // for src cast
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize]; // need splitM * 64
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitSize + tiling.splitM * FLOAT_REPEAT_SIZE];
    const LocalTensor<float>& tmpBuffer3 =
        workLocal[tiling.splitSize + tiling.splitM * FLOAT_REPEAT_SIZE + tiling.reduceSize];

    Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
    Cast(tmpBuffer3, inMaxTensor[offset2], RoundMode::CAST_NONE, reduceSize);

    ReduceMaxImpl(tmpBuffer2, tmpBuffer0, tmpBuffer1, reduceParam);
    Max(tmpBuffer2, tmpBuffer2, tmpBuffer3, reduceSize); // new_max
    Cast(maxTensor[offset2], tmpBuffer2, RoundMode::CAST_ROUND, reduceSize);

    Sub(tmpBuffer3, tmpBuffer3, tmpBuffer2, reduceSize);
    Exp(tmpBuffer3, tmpBuffer3, reduceSize); // exp(inmax - new_max)

    Cast(expMaxTensor[offset2], tmpBuffer3, RoundMode::CAST_ROUND, reduceSize);

    SubNDImpl(tmpBuffer0, tmpBuffer0, tmpBuffer2, reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);
    Exp(tmpBuffer0, tmpBuffer0, splitSize); // exp(x - new_max)
    Cast(dst[offset1], tmpBuffer0, RoundMode::CAST_ROUND, splitSize);

    ReduceSumImpl(tmpBuffer2, tmpBuffer0, tmpBuffer1, reduceParam);

    Cast(tmpBuffer1, inSumTensor[offset2], RoundMode::CAST_NONE, reduceSize);
    Mul(tmpBuffer1, tmpBuffer3, tmpBuffer1, reduceSize);
    Add(tmpBuffer1, tmpBuffer1, tmpBuffer2, reduceSize);
    Cast(sumTensor[offset2], tmpBuffer1, RoundMode::CAST_ROUND, reduceSize);
}
__aicore__ inline void SoftmaxFlashV2UpdateImpl(const LocalTensor<float>& dst, const LocalTensor<float>& expSumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<float>& src, const LocalTensor<float>& expMaxTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor,
    const LocalTensor<float>& workLocal, const ReduceLastND& reduceParam, const SoftMaxTiling& tiling,
    const uint32_t& offset1, const uint32_t& offset2, const uint32_t& splitSize, const uint32_t& reduceSize)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.reduceSize]; // tiling.splitM * FLOAT_REPEAT_SIZE

    ReduceMaxImpl(tmpBuffer0, src[offset1], tmpBuffer1, reduceParam);
    Max(tmpBuffer0, inMaxTensor[offset2], tmpBuffer0, reduceSize);

    Sub(tmpBuffer1, inMaxTensor[offset2], tmpBuffer0, reduceSize);
    Exp(expMaxTensor[offset2], tmpBuffer1, reduceSize); // exp(inmax - new_max)
    DataCopy(maxTensor[offset2], tmpBuffer0, reduceSize);

    SubNDImpl(dst[offset1], src[offset1], maxTensor[offset2], reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);
    Exp(dst[offset1], dst[offset1], splitSize); // exp(x - new_max)
    ReduceSumImpl(tmpBuffer0, dst[offset1], tmpBuffer1, reduceParam);

    Mul(expSumTensor[offset2], expMaxTensor[offset2], inExpSumTensor[offset2], reduceSize);
    Add(expSumTensor[offset2], expSumTensor[offset2], tmpBuffer0, reduceSize);
}


template <typename T1, typename T2, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV2Update(const LocalTensor<T1>& dst, const LocalTensor<T2>& expSumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<T1>& expMaxTensor,
    const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    ReduceLastND reduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
        tiling.splitK, tiling.reduceM,     tiling.reduceK };
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitSize = tiling.splitSize;
    uint32_t reduceSize = tiling.reduceSize;
    for (uint32_t i = 0; i <= tiling.rangeM; i++) {
        SoftmaxFlashV2UpdateImpl(dst, expSumTensor, maxTensor, src, expMaxTensor, inExpSumTensor, inMaxTensor,
            workLocal, reduceParam, tiling, offset1, offset2, splitSize, reduceSize);
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

template <typename T1, typename T2, bool isUpdate = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV2PostProcess(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& expSumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
    const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    if constexpr (!isUpdate) {
        SoftmaxFlashV2NoUpdate<T1, T2, isBasicBlock>(dstTensor, expSumTensor, maxTensor, srcTensor, workLocal,
            originalSrcShape, tiling);
    } else {
        SoftmaxFlashV2Update<T1, T2, isBasicBlock>(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, workLocal, originalSrcShape, tiling);
    }
}
template <typename T1, typename T2, bool isUpdate = false, bool isBasicBlock = false>
__aicore__ inline void SoftMaxFlashV2NZImpl(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
    const LocalTensor<T2>& inSumTensor, const LocalTensor<T2>& inMaxTensor, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "softmaxflashv2 format NZ is not supported on current device!"); });
}
template <typename T1, typename T2, bool isUpdate = false, bool isBasicBlock = false>
__aicore__ inline void SoftMaxFlashV2NZImpl(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "softmaxflashv2 format NZ is not supported on current device!"); });
}
template <typename T1, typename T2, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV2NDImpl(const LocalTensor<half>& dst, const LocalTensor<float>& expSumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "softmaxflashv2 current data type is not supported on current device!"); });
}

template <typename T1, typename T2, bool isUpdate = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashV2M1PostProcess(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& expSumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
    const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "softmaxflashv2 current data struct is not supported on current device!"); });
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_V300_SOFTMAX_FLASHV2_IMPL_H