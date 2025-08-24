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
 * \file softmax_base_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_BASE_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_BASE_IMPL_H

#if __CCE_AICORE__ == 300
#include "regbase/v300/softmax_impl.h"
#elif __CCE_AICORE__ == 220
#include "membase/v220/softmax_impl.h"
#elif __CCE_AICORE__ == 200
#include "membase/v200/softmax_impl.h"
#endif

namespace AscendC {
template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
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
    // when the shape is changed, need recalculate the softmax's tiling
    if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
        SoftMaxTiling newTiling = tiling;
        SoftMaxTilingFunc(workLocal.GetSize(), { srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k },
            newTiling, sizeof(T), sizeof(float), isBasicBlock);
        SoftMaxNDImpl<T, isReuseSource, isBasicBlock, config>(dst, src, workLocal, originalSrcShape, newTiling);
    } else {
        SoftMaxNDImpl<T, isReuseSource, isBasicBlock, config>(dst, src, workLocal, originalSrcShape, tiling);
    }
}
template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    SoftMaxImpl<T, isReuseSource, isBasicBlock, config>(dst, src, workLocal, tiling, softmaxShapeInfo);
}
template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SoftMaxImpl<T, isReuseSource, isBasicBlock, config>(dst, src, workLocal, tiling, softmaxShapeInfo);
}

template <typename T1, typename T2, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
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
        // when the shape is changed, need recalculate the softmax's tiling
        if (unlikely(srcNDinfo.k != tiling.srcK || originalSrcShape.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxTilingFunc(workLocal.GetSize(), { srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k },
                newTiling, sizeof(T1), sizeof(T2), false, isDataFormatNZ);
            SoftMaxNZImpl<T1, T2, isBasicBlock>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, newTiling);
        } else {
            SoftMaxNZImpl<T1, T2, isBasicBlock>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
        }
    } else {
        // when the shape is changed, need recalculate the softmax's tiling
        if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxTilingFunc(workLocal.GetSize(), { srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k },
                newTiling, sizeof(T1), sizeof(T2), isBasicBlock);
            SoftMaxNDImpl<T1, T2, isBasicBlock, config>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape,
                newTiling);
        } else {
            SoftMaxNDImpl<T1, T2, isBasicBlock, config>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape,
                tiling);
        }
    }
}

template <typename T1, typename T2, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    SoftMaxImpl<T1, T2, isBasicBlock, isDataFormatNZ, config>(dst, sumTensor, maxTensor, src, workLocal, tiling,
        softmaxShapeInfo);
}

template <typename T1, typename T2, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SoftMaxImpl<T1, T2, isBasicBlock, isDataFormatNZ, config>(dst, sumTensor, maxTensor, src, workLocal, tiling,
        softmaxShapeInfo);
}

template <typename T1, typename T2, uint8_t stepSizeMode = 0>
__aicore__ inline bool AdjustSoftMaxResNZImpl(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
    const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    uint32_t floatStepSize = ONE_BLK_FLOAT_NUM;
    uint32_t halfStepSize = ONE_BLK_HALF_NUM;
    if constexpr (stepSizeMode) {
        floatStepSize = 1;
        halfStepSize = 1;
    }

    bool isUpdateNeedCheck = false;
    const uint32_t splitNZBlockCount = softmaxShapeInfo.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    SetVectorMask<float>(SOFTMAX_SHAPE_NZ_BASIC_COUNT);
    for (uint32_t j = 0; j < softmaxShapeInfo.srcM; j++) {
        uint32_t offset = j * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        uint32_t splitCount = softmaxShapeInfo.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        if constexpr (sizeof(T2) == sizeof(float)) {
            T2 maxValue = maxTensor.GetValue(j * floatStepSize);
            uint32_t checkValue = *reinterpret_cast<uint32_t*>(&maxValue);
            if (checkValue == from) {
                for (uint32_t k = 0; k < splitNZBlockCount; k++) {
                    Duplicate<T1, false>(softMaxRes[offset + splitCount * k], to, MASK_PLACEHOLDER, 1, 1,
                        DEFAULT_REPEAT_STRIDE);
                }
                isUpdateNeedCheck = true;
            }
        } else {
            T2 maxValue = maxTensor.GetValue(j * halfStepSize);
            uint16_t checkValue = *reinterpret_cast<uint16_t*>(&maxValue);
            if (checkValue == (uint16_t)from) {
                for (uint32_t k = 0; k < splitNZBlockCount; k++) {
                    Duplicate<T1, false>(softMaxRes[offset + splitCount * k], to, MASK_PLACEHOLDER, 1, 1,
                        DEFAULT_REPEAT_STRIDE);
                }
                isUpdateNeedCheck = true;
            }
        }
    }
    ResetMask();
    return isUpdateNeedCheck;
}

template <typename T1, typename T2, uint8_t stepSizeMode = 0>
__aicore__ inline bool AdjustSoftMaxResNDImpl(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
    const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    uint32_t floatStepSize = ONE_BLK_FLOAT_NUM;
    uint32_t halfStepSize = ONE_BLK_HALF_NUM;
    if constexpr (stepSizeMode) {
        floatStepSize = 1;
        halfStepSize = 1;
    }

    bool isUpdateNeedCheck = false;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, softmaxShapeInfo.srcK);
    for (uint32_t j = 0; j < softmaxShapeInfo.srcM; j++) {
        if constexpr (sizeof(T2) == sizeof(float)) {
            T2 maxValue = maxTensor.GetValue(j * floatStepSize);
            uint32_t checkValue = *reinterpret_cast<uint32_t*>(&maxValue);
            if (checkValue == from) {
                Duplicate<T1, false>(softMaxRes[j * softmaxShapeInfo.srcK], to, MASK_PLACEHOLDER, 1, 1,
                    DEFAULT_REPEAT_STRIDE);
                isUpdateNeedCheck = true;
            }
        } else {
            T2 maxValue = maxTensor.GetValue(j * halfStepSize);
            uint16_t checkValue = *reinterpret_cast<uint16_t*>(&maxValue);
            if (checkValue == (uint16_t)from) {
                Duplicate<T1, false>(softMaxRes[j * softmaxShapeInfo.srcK], to, MASK_PLACEHOLDER, 1, 1,
                    DEFAULT_REPEAT_STRIDE);
                isUpdateNeedCheck = true;
            }
        }
    }
    SetMaskNorm();
    ResetMask();
    return isUpdateNeedCheck;
}

template <typename T1, typename T2, bool isDataFormatNZ = false, uint8_t stepSizeMode = 0>
__aicore__ inline bool AdjustSoftMaxResImpl(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
    const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    SetMaskNorm();
    ResetMask();
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    bool isUpdateNeedCheck = false;
    if constexpr (isDataFormatNZ) {
        isUpdateNeedCheck = AdjustSoftMaxResNZImpl<T1, T2, stepSizeMode>(softMaxRes, maxTensor, from, to, softmaxShapeInfo);
    } else {
        isUpdateNeedCheck = AdjustSoftMaxResNDImpl<T1, T2, stepSizeMode>(softMaxRes, maxTensor, from, to, softmaxShapeInfo);
    }
    return isUpdateNeedCheck;
}
}

#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_BASE_IMPL_H