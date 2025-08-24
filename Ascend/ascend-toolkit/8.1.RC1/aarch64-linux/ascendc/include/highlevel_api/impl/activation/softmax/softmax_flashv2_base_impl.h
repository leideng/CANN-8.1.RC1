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
 * \file softmax_flashv2_base_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_BASE_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_BASE_IMPL_H

#if __CCE_AICORE__ == 300
#include "regbase/v300/softmax_flashv2_impl.h"
#elif __CCE_AICORE__ == 220
#include "membase/v220/softmax_flashv2_impl.h"
#elif __CCE_AICORE__ == 200
#include "membase/v200/softmax_flashv2_impl.h"
#endif

namespace AscendC {
__aicore__ inline constexpr SoftMaxTiling SoftMaxFlashV2TilingFuncImpl(const uint32_t srcM, const uint32_t srcK,
    const uint32_t dataTypeSize1, const uint32_t dataTypeSize2, const uint32_t localWorkSpaceSize,
    const bool isUpdate = false, const bool isBasicBlock = false, const bool isDataFormatNZ = false,
    const bool isFlashOutputBrc = false)
{
    SoftMaxTiling softmaxTiling;
    const uint32_t elementNumPerBlk = ONE_BLK_SIZE / dataTypeSize2;
    softmaxTiling.srcM = srcM;
    softmaxTiling.srcK = srcK;
    softmaxTiling.srcSize = srcM * srcK;

    softmaxTiling.outMaxM = srcM;
    softmaxTiling.outMaxK = elementNumPerBlk;
    softmaxTiling.outMaxSize = srcM * elementNumPerBlk;

    if (isDataFormatNZ) {
        softmaxTiling.reduceM = localWorkSpaceSize / (SOFTMAX_SHAPE_NZ_BASIC_COUNT * SOFTMAX_NZ_TILING_NEEDBLOCK + srcK);
    } else {
        if (isBasicBlock && srcK % FLOAT_REPEAT_SIZE == 0 && srcM % SOFTMAX_BASIC_TILE_NUM == 0) {
            softmaxTiling.reduceM =
                CalculateNDSplitM(localWorkSpaceSize, dataTypeSize1, elementNumPerBlk, { srcM, srcK }, isBasicBlock);
        } else {
            softmaxTiling.reduceM = (dataTypeSize1 == B16_BYTE_SIZE) ?
                localWorkSpaceSize / (SOFTMAX_COMPUTE_DIM * elementNumPerBlk + srcK + FLOAT_REPEAT_SIZE) :
                localWorkSpaceSize / (elementNumPerBlk + FLOAT_REPEAT_SIZE);
        }
    }

    uint32_t softmaxBasicTileNum = SOFTMAX_BASIC_TILE_NUM;
    if (isFlashOutputBrc && dataTypeSize1 == B16_BYTE_SIZE) {
        softmaxBasicTileNum = HALF_NUM_PER_BLK;
    }

    if (softmaxTiling.reduceM < srcM && softmaxTiling.reduceM > softmaxBasicTileNum) {
        softmaxTiling.reduceM = softmaxTiling.reduceM / softmaxBasicTileNum * softmaxBasicTileNum;
    }
    softmaxTiling.reduceM = softmaxTiling.reduceM < srcM ? softmaxTiling.reduceM : srcM;

    softmaxTiling.reduceK = elementNumPerBlk;
    softmaxTiling.reduceSize = softmaxTiling.reduceM * elementNumPerBlk;

    softmaxTiling.splitM = softmaxTiling.reduceM;
    softmaxTiling.splitK = srcK;
    softmaxTiling.splitSize = softmaxTiling.reduceM * srcK;

    softmaxTiling.rangeM = srcM / softmaxTiling.reduceM;
    softmaxTiling.tailM = srcM % softmaxTiling.reduceM;

    softmaxTiling.tailSplitSize = softmaxTiling.tailM * srcK;
    softmaxTiling.tailReduceSize = softmaxTiling.tailM * elementNumPerBlk;

    if (isFlashOutputBrc && (softmaxTiling.rangeM > MIN_BLOCK_LEN || softmaxTiling.tailM != 0)) {
        ASCENDC_ASSERT((softmaxTiling.reduceM % (ONE_BLK_SIZE / dataTypeSize1) == 0), {printf("[ERROR] "
            "When dataTypeSize1(%d) is float(or half), softmaxTiling.reduceM(%d) must be a multiple of 8(or 16), "
            "Adjust the input parameter -> localWorkSpaceSize.\n", dataTypeSize1, softmaxTiling.reduceM);});
    }
    return softmaxTiling;
}

template <typename T1, typename T2, bool isUpdate, bool isBasicBlock, bool isDataFormatNZ, const SoftmaxConfig& config>
__aicore__ inline SoftMaxTiling SoftmaxFlashV2UpdateTilingImpl(const LocalTensor<T1>& srcTensor, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    if constexpr (isDataFormatNZ) {
        LastAxisShapeND srcNDinfo = { softmaxShapeInfo.srcM, softmaxShapeInfo.srcK };
        if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
            ShapeInfo srcShape = srcTensor.GetShapeInfo();
            srcNDinfo = GetLastAxisShapeND(srcShape);
        }
        uint32_t workLocalSize = workLocal.GetSize();
        return SoftMaxFlashV2TilingFuncImpl(srcNDinfo.m, srcNDinfo.k, sizeof(T1), sizeof(T2), workLocalSize, isUpdate, false, true);
    } else {
        if constexpr (!config.isCheckTiling) {
            return tiling;
        }

        LastAxisShapeND srcNDinfo = { softmaxShapeInfo.srcM, softmaxShapeInfo.srcK };
        if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
            ShapeInfo srcShape = srcTensor.GetShapeInfo();
            srcNDinfo = GetLastAxisShapeND(srcShape);
        }

        if (srcNDinfo.m == tiling.srcM && srcNDinfo.k == tiling.srcK) {
            return tiling;
        }

        SoftMaxTiling softmaxTiling;
        uint32_t workLocalSize = workLocal.GetSize();

        if constexpr (config.mode == SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC) {
            softmaxTiling = SoftMaxFlashV2TilingFuncImpl(srcNDinfo.m, srcNDinfo.k, sizeof(T1), sizeof(T2), workLocalSize, isUpdate, isBasicBlock, false, true);
        } else {
            softmaxTiling = SoftMaxFlashV2TilingFuncImpl(srcNDinfo.m, srcNDinfo.k, sizeof(T1), sizeof(T2), workLocalSize, isUpdate, isBasicBlock);
        }
        return softmaxTiling;
    }
}

template <typename T1, typename T2, bool isUpdate, bool isBasicBlock, bool isDataFormatNZ, const SoftmaxConfig& config>
__aicore__ inline void SoftmaxFlashV2Impl(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
    const LocalTensor<T2>& inSumTensor, const LocalTensor<T2>& inMaxTensor, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    SetMaskNorm();
    ResetMask();

    SoftMaxTiling newTiling = SoftmaxFlashV2UpdateTilingImpl<T1, T2, isUpdate, isBasicBlock, isDataFormatNZ, config>(
        srcTensor, workLocal, tiling, softmaxShapeInfo);

    LastAxisShapeND originalSrcShape = { softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK };
    if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
        ShapeInfo srcShape = srcTensor.GetShapeInfo();
        originalSrcShape = GetLastAxisOriginShapeND(srcShape);
    }

    if constexpr (isDataFormatNZ) {
        SoftMaxFlashV2NZImpl<T1, T2, isUpdate, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor,
            expMaxTensor, inSumTensor, inMaxTensor, workLocal, originalSrcShape, newTiling);
    } else if constexpr (config.mode == SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC) {
        SoftmaxFlashV2M1PostProcess<T1, T2, isUpdate, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor,
            expMaxTensor, inSumTensor, inMaxTensor, workLocal, originalSrcShape, newTiling);
    } else {
        SoftmaxFlashV2PostProcess<T1, T2, isUpdate, isBasicBlock, config>(dstTensor, sumTensor, maxTensor, srcTensor,
            expMaxTensor, inSumTensor, inMaxTensor, workLocal, originalSrcShape, newTiling);
    }
}

template <typename T1, typename T2, bool isUpdate, bool isBasicBlock, bool isDataFormatNZ, const SoftmaxConfig& config>
__aicore__ inline void SoftmaxFlashV2Impl(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
    const LocalTensor<T2>& inSumTensor, const LocalTensor<T2>& inMaxTensor, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    SoftmaxFlashV2Impl<T1, T2, isUpdate, isBasicBlock, isDataFormatNZ, config>(dstTensor, sumTensor, maxTensor,
        srcTensor, expMaxTensor, inSumTensor, inMaxTensor, workLocal, tiling, softmaxShapeInfo);
}

template <typename T1, typename T2, bool isUpdate, bool isBasicBlock, bool isDataFormatNZ, const SoftmaxConfig& config>
__aicore__ inline void SoftmaxFlashV2Impl(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
    const LocalTensor<T2>& inSumTensor, const LocalTensor<T2>& inMaxTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    if constexpr (isDataFormatNZ) { // check nz format tmpbuffer size
        LastAxisShapeND srcNDinfo = { softmaxShapeInfo.srcM, softmaxShapeInfo.srcK };
        if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
            ShapeInfo srcShape = srcTensor.GetShapeInfo();
            srcNDinfo = GetLastAxisShapeND(srcShape);
        }
        uint32_t workLocalSize = workLocal.GetSize();
        if (workLocalSize < SOFTMAX_SHAPE_NZ_BASIC_COUNT * SOFTMAX_NZ_TILING_NEEDBLOCK + srcNDinfo.k) {
            PopStackBuffer<float, TPosition::LCM>(workLocal);
        }
    }

    SoftmaxFlashV2Impl<T1, T2, isUpdate, isBasicBlock, isDataFormatNZ, config>(dstTensor, sumTensor, maxTensor,
        srcTensor, expMaxTensor, inSumTensor, inMaxTensor, workLocal, tiling, softmaxShapeInfo);
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_BASE_IMPL_H