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
 * \file simple_softmax_base_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_BASE_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_BASE_IMPL_H

#if __CCE_AICORE__ == 300
#include "regbase/v300/simple_softmax_impl.h"
#elif __CCE_AICORE__ == 220
#include "membase/v220/simple_softmax_impl.h"
#elif __CCE_AICORE__ == 200
#include "membase/v200/simple_softmax_impl.h"
#endif

namespace AscendC {
template <typename T1, typename T2, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& inSumTensor,
    const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    SetMaskNorm();
    ResetMask();
    ShapeInfo srcShape = src.GetShapeInfo();
    LastAxisShapeND srcNDinfo;
    LastAxisShapeND originalSrcShape;
    if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
        srcNDinfo = GetLastAxisShapeND(srcShape);
        originalSrcShape = GetLastAxisOriginShapeND(srcShape);
    } else {
        srcNDinfo = { softmaxShapeInfo.srcM, softmaxShapeInfo.srcK };
        originalSrcShape = { softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK };
    }
    if constexpr (isDataFormatNZ) {
        if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxTilingFunc(workLocal.GetSize(), { srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k },
                newTiling, sizeof(T1), sizeof(T2), false, isDataFormatNZ);
            SimpleSoftMaxNZImpl<T1, T2>(dst, inSumTensor, inMaxTensor, src, workLocal, newTiling);
        } else {
            SimpleSoftMaxNZImpl<T1, T2>(dst, inSumTensor, inMaxTensor, src, workLocal, tiling);
        }
    } else {
        if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxTilingFunc(workLocal.GetSize(), { srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k },
                newTiling, sizeof(T1), sizeof(T2), isBasicBlock);
            SimpleSoftMaxNDImpl<T1, isBasicBlock, config>(dst, inSumTensor, inMaxTensor, src, workLocal, newTiling);
        } else {
            SimpleSoftMaxNDImpl<T1, isBasicBlock, config>(dst, inSumTensor, inMaxTensor, src, workLocal, tiling);
        }
    }
}

template <typename T1, typename T2, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& inSumTensor,
    const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& src, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    SimpleSoftMaxImpl<T1, T2, isBasicBlock, isDataFormatNZ, config>(dst, inSumTensor, inMaxTensor, src, workLocal,
        tiling, softmaxShapeInfo);
}

template <typename T1, typename T2, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& inSumTensor,
    const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& src, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SimpleSoftMaxImpl<T1, T2, isBasicBlock, isDataFormatNZ, config>(dst, inSumTensor, inMaxTensor, src, workLocal,
        tiling, softmaxShapeInfo);
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_BASE_IMPL_H