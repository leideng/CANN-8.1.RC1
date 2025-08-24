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
 * \file softmax_grad_base_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_GRAD_BASE_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_GRAD_BASE_IMPL_H

#if __CCE_AICORE__ == 300
#include "regbase/v300/softmax_grad_impl.h"
#elif __CCE_AICORE__ == 220
#include "membase/v220/softmax_grad_impl.h"
#elif __CCE_AICORE__ == 200
#include "membase/v200/softmax_grad_impl.h"
#endif

namespace AscendC {
template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradFrontImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    ShapeInfo srcShape = srcTensor.GetShapeInfo();
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
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
            SoftMaxGradTilingFunc(workLocal.GetSize(), srcNDinfo, newTiling, elementNumPerBlk, true, false, true);
            SoftmaxGradFrontNZImpl(dstTensor, gradTensor, srcTensor, workLocal, originalSrcShape, newTiling);
        } else {
            SoftmaxGradFrontNZImpl(dstTensor, gradTensor, srcTensor, workLocal, originalSrcShape, tiling);
        }
    } else {
        if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxGradTilingFunc(workLocal.GetSize(), srcNDinfo, newTiling, elementNumPerBlk, true, isBasicBlock);
            SoftmaxGradFrontNDImpl<T, isBasicBlock>(dstTensor, gradTensor, srcTensor, workLocal, newTiling,
                originalSrcShape);
        } else {
            SoftmaxGradFrontNDImpl<T, isBasicBlock>(dstTensor, gradTensor, srcTensor, workLocal, tiling,
                originalSrcShape);
        }
    }
}

template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradFrontImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    SoftmaxGradFrontImpl<T, isBasicBlock, isDataFormatNZ>(dstTensor, gradTensor, srcTensor, workLocal, tiling,
        softmaxShapeInfo);
}

template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradFrontImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SoftmaxGradFrontImpl<T, isBasicBlock, isDataFormatNZ>(dstTensor, gradTensor, srcTensor, workLocal, tiling,
        softmaxShapeInfo);
}

template <typename T, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling, bool isFront,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    ShapeInfo srcShape = srcTensor.GetShapeInfo();
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);

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
            SoftMaxGradTilingFunc(workLocal.GetSize(), srcNDinfo, newTiling, elementNumPerBlk, isFront, false, true);
            SoftmaxGradNZImpl(dstTensor, gradTensor, srcTensor, workLocal, originalSrcShape, newTiling, isFront);
        } else {
            SoftmaxGradNZImpl(dstTensor, gradTensor, srcTensor, workLocal, originalSrcShape, tiling, isFront);
        }
    } else {
        if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxGradTilingFunc(workLocal.GetSize(), srcNDinfo, newTiling, elementNumPerBlk, isFront, false);
            SoftmaxGradPostProcess<T>(dstTensor, gradTensor, srcTensor, workLocal, newTiling, originalSrcShape,
                isFront);
        } else {
            SoftmaxGradPostProcess<T>(dstTensor, gradTensor, srcTensor, workLocal, tiling, originalSrcShape, isFront);
        }
    }
}

template <typename T, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, bool isFront,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    SoftmaxGradImpl<T, isDataFormatNZ>(dstTensor, gradTensor, srcTensor, workLocal, tiling, isFront, softmaxShapeInfo);
}
template <typename T, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
    bool isFront, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SoftmaxGradImpl<T, isDataFormatNZ>(dstTensor, gradTensor, srcTensor, workLocal, tiling, isFront, softmaxShapeInfo);
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_GRAD_BASE_IMPL_H