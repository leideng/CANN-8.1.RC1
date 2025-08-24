/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file kernel_utils_struct_confusion_pad.h
 * \brief
 */
#ifndef ASCENDC_MODULE_UTILS_STRUCT_CONFUSION_PAD_H
#define ASCENDC_MODULE_UTILS_STRUCT_CONFUSION_PAD_H
#include "utils/kernel_utils_mode.h"

namespace AscendC {
struct ConfusionTranspose2NZ012NTiling {
    __aicore__ ConfusionTranspose2NZ012NTiling()
    {
        blockSize = 0;
        shapeB = 0;
        shapeN = 0;
        hnDiv = 0;
        blockNum = 0;
        shapeH = 0;
        hBlockNum = 0;
        sBlockNum = 0;
        alignH = 0;
        alignS = 0;
        hnDivBlockNum = 0;
        alignHnDiv = 0;
        gap = 0;
        alignsBlockCube = 0;
        prehBlockNum = 0;
        dstBatchOffset = 0;
        srcBatchOffset = 0;
    }

    __aicore__ ConfusionTranspose2NZ012NTiling(uint32_t blockSizeIn, uint32_t shapeBIn, uint32_t shapeNIn,
        uint32_t hnDivIn, uint32_t blockNumIn, uint32_t shapeHIn, uint32_t hBlockNumIn, uint32_t sBlockNumIn,
        uint32_t alignHIn, uint32_t alignSIn, uint32_t hnDivBlockNumIn, uint32_t alignHnDivIn, uint32_t gapIn,
        uint32_t alignsBlockCubeIn, uint32_t prehBlockNumIn, uint32_t dstBatchOffsetIn, uint32_t srcBatchOffsetIn)
    {
        blockSize = blockSizeIn;
        shapeB = shapeBIn;
        shapeN = shapeNIn;
        hnDiv = hnDivIn;
        blockNum = blockNumIn;
        shapeH = shapeHIn;
        hBlockNum = hBlockNumIn;
        sBlockNum = sBlockNumIn;
        alignH = alignHIn;
        alignS = alignSIn;
        hnDivBlockNum = hnDivBlockNumIn;
        alignHnDiv = alignHnDivIn;
        gap = gapIn;
        alignsBlockCube = alignsBlockCubeIn;
        prehBlockNum = prehBlockNumIn;
        dstBatchOffset = dstBatchOffsetIn;
        srcBatchOffset = srcBatchOffsetIn;
    }
    uint32_t blockSize = 0;
    uint32_t shapeB = 0;
    uint32_t shapeN = 0;
    uint32_t hnDiv = 0;
    uint32_t blockNum = 0;
    uint32_t shapeH = 0;
    uint32_t hBlockNum = 0;
    uint32_t sBlockNum = 0;
    uint32_t alignH = 0;
    uint32_t alignS = 0;
    uint32_t hnDivBlockNum = 0;
    uint32_t alignHnDiv = 0;
    uint32_t gap = 0;
    uint32_t alignsBlockCube = 0;
    uint32_t prehBlockNum = 0;
    uint32_t dstBatchOffset = 0;
    uint32_t srcBatchOffset = 0;
};

struct ConfusionTranspose2ND012NTiling {
    __aicore__ ConfusionTranspose2ND012NTiling()
    {
        blockSize = 0;
        shapeB = 0;
        shapeN = 0;
        hnDiv = 0;
        shapeH = 0;
        hBlockNum = 0;
        sBlockNum = 0;
        hnDivBlockNum = 0;
        alignHnDiv = 0;
        gap = 0;
        alignsCube = 0;
        prehBlockNum = 0;
        alignsMulAlignHnDiv = 0;
        alignHnDivCube = 0;
        alignHnDivBlockSize = 0;
        dstBatchOffset = 0;
        srcBatchOffset = 0;
        blockNum = 0;
    }

    __aicore__ ConfusionTranspose2ND012NTiling(uint32_t blockSizeIn, uint32_t shapeBIn, uint32_t shapeNIn,
        uint32_t hnDivIn, uint32_t shapeHIn, uint32_t hBlockNumIn, uint32_t sBlockNumIn, uint32_t hnDivBlockNumIn,
        uint32_t alignHnDivIn, uint32_t gapIn, uint32_t alignsCubeIn, uint32_t prehBlockNumIn,
        uint32_t alignsMulAlignHnDivIn, uint32_t alignHnDivCubeIn, uint32_t alignHnDivBlockSizeIn,
        uint32_t dstBatchOffsetIn, uint32_t srcBatchOffsetIn, uint32_t blockNumIn)
    {
        blockSize = blockSizeIn;
        shapeB = shapeBIn;
        shapeN = shapeNIn;
        hnDiv = hnDivIn;
        shapeH = shapeHIn;
        hBlockNum = hBlockNumIn;
        sBlockNum = sBlockNumIn;
        hnDivBlockNum = hnDivBlockNumIn;
        alignHnDiv = alignHnDivIn;
        gap = gapIn;
        alignsCube = alignsCubeIn;
        prehBlockNum = prehBlockNumIn;
        alignsMulAlignHnDiv = alignsMulAlignHnDivIn;
        alignHnDivCube = alignHnDivCubeIn;
        alignHnDivBlockSize = alignHnDivBlockSizeIn;
        dstBatchOffset = dstBatchOffsetIn;
        srcBatchOffset = srcBatchOffsetIn;
        blockNum = blockNumIn;
    }
    uint32_t blockSize = 0;
    uint32_t shapeB = 0;
    uint32_t shapeN = 0;
    uint32_t hnDiv = 0;
    uint32_t shapeH = 0;
    uint32_t hBlockNum = 0;
    uint32_t sBlockNum = 0;
    uint32_t hnDivBlockNum = 0;
    uint32_t alignHnDiv = 0;
    uint32_t gap = 0;
    uint32_t alignsCube = 0;
    uint32_t prehBlockNum = 0;
    uint32_t alignsMulAlignHnDiv = 0;
    uint32_t alignHnDivCube = 0;
    uint32_t alignHnDivBlockSize = 0;
    uint32_t dstBatchOffset = 0;
    uint32_t srcBatchOffset = 0;
    uint32_t blockNum = 0;
};

struct ConfusionTranspose012Tiling {
    __aicore__ ConfusionTranspose012Tiling()
    {
        blockSize = 0;
        shapeB = 0;
        shapeN = 0;
        hnDiv = 0;
        shapeH = 0;
        hBlockNum = 0;
        sBlockNum = 0;
        hnDivBlockNum = 0;
        alignH = 0;
        alignsCube = 0;
        alignhBlockCube = 0;
        blockSizeMulAlignH = 0;
        srcBatchOffset = 0;
        dstBatchOffset = 0;
        blockNum = 0;
    }

    __aicore__ ConfusionTranspose012Tiling(uint32_t blockSizeIn, uint32_t shapeBIn, uint32_t shapeNIn, uint32_t hnDivIn,
        uint32_t shapeHIn, uint32_t hBlockNumIn, uint32_t sBlockNumIn, uint32_t hnDivBlockNumIn, uint32_t alignHIn,
        uint32_t alignsCubeIn, uint32_t alignhBlockCubeIn, uint32_t blockSizeMulAlignHIn, uint32_t srcBatchOffsetIn,
        uint32_t dstBatchOffsetIn, uint32_t blockNumIn)
    {
        blockSize = blockSizeIn;
        shapeB = shapeBIn;
        shapeN = shapeNIn;
        hnDiv = hnDivIn;
        shapeH = shapeHIn;
        hBlockNum = hBlockNumIn;
        sBlockNum = sBlockNumIn;
        hnDivBlockNum = hnDivBlockNumIn;
        alignH = alignHIn;
        alignsCube = alignsCubeIn;
        alignhBlockCube = alignhBlockCubeIn;
        blockSizeMulAlignH = blockSizeMulAlignHIn;
        srcBatchOffset = srcBatchOffsetIn;
        dstBatchOffset = dstBatchOffsetIn;
        blockNum = blockNumIn;
    }
    uint32_t blockSize = 0;
    uint32_t shapeB = 0;
    uint32_t shapeN = 0;
    uint32_t hnDiv = 0;
    uint32_t shapeH = 0;
    uint32_t hBlockNum = 0;
    uint32_t sBlockNum = 0;
    uint32_t hnDivBlockNum = 0;
    uint32_t alignH = 0;
    uint32_t alignsCube = 0;
    uint32_t alignhBlockCube = 0;
    uint32_t blockSizeMulAlignH = 0;
    uint32_t srcBatchOffset = 0;
    uint32_t dstBatchOffset = 0;
    uint32_t blockNum = 0;
};

struct ConfusionTransposeOnlyTiling {
    __aicore__ ConfusionTransposeOnlyTiling()
    {
        blockSize = 0;
        height = 0;
        width = 0;
        highBlock = 0;
        stride = 0;
        repeat = 0;
    }

    __aicore__ ConfusionTransposeOnlyTiling(uint32_t blockSizeIn, uint32_t heightIn, uint32_t widthIn,
        uint32_t highBlockIn, uint32_t strideIn, uint32_t repeatIn)
    {
        blockSize = blockSizeIn;
        height = heightIn;
        width = widthIn;
        highBlock = highBlockIn;
        stride = strideIn;
        repeat = repeatIn;
    }
    uint32_t blockSize = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t highBlock = 0;
    uint32_t stride = 0;
    uint32_t repeat = 0;
};

struct ConfusionTranspose0213Tiling {
    __aicore__ ConfusionTranspose0213Tiling()
    {
        blockSize = 0;
        shapeB = 0;
        shapeA1 = 0;
        alignA3 = 0;
        alignA2 = 0;
        widthTiling = 0;
        newPopSize = 0;
        newPopH = 0;
        needSize = 0;
        mainBlocks = 0;
        tailSize = 0;
        alignA2MulAlignA3 = 0;
        batchOffset = 0;
        alignA3MulA1 = 0;
        shapeA1BlockCube = 0;
        mainOffset = 0;
    }

    __aicore__ ConfusionTranspose0213Tiling(uint32_t blockSizeIn, uint32_t shapeBIn, uint32_t shapeA1In,
        uint32_t alignA3In, uint32_t alignA2In, uint32_t widthTilingIn, uint32_t newPopSizeIn, uint32_t newPopHIn,
        uint32_t needSizeIn, uint32_t mainBlocksIn, uint32_t tailSizeIn, uint32_t alignA2MulAlignA3In,
        uint32_t batchOffsetIn, uint32_t alignA3MulA1In, uint32_t shapeA1BlockCubeIn, uint32_t mainOffsetIn)
    {
        blockSize = blockSizeIn;
        shapeB = shapeBIn;
        shapeA1 = shapeA1In;
        alignA3 = alignA3In;
        alignA2 = alignA2In;
        widthTiling = widthTilingIn;
        newPopSize = newPopSizeIn;
        newPopH = newPopHIn;
        needSize = needSizeIn;
        mainBlocks = mainBlocksIn;
        tailSize = tailSizeIn;
        alignA2MulAlignA3 = alignA2MulAlignA3In;
        batchOffset = batchOffsetIn;
        alignA3MulA1 = alignA3MulA1In;
        shapeA1BlockCube = shapeA1BlockCubeIn;
        mainOffset = mainOffsetIn;
    }
    uint32_t blockSize = 0;
    uint32_t shapeB = 0;
    uint32_t shapeA1 = 0;
    uint32_t alignA3 = 0;
    uint32_t alignA2 = 0;
    uint32_t widthTiling = 0;
    uint32_t newPopSize = 0;
    uint32_t newPopH = 0;
    uint32_t needSize = 0;
    uint32_t mainBlocks = 0;
    uint32_t tailSize = 0;
    uint32_t alignA2MulAlignA3 = 0;
    uint32_t batchOffset = 0;
    uint32_t alignA3MulA1 = 0;
    uint32_t shapeA1BlockCube = 0;
    uint32_t mainOffset = 0;
};

struct IntriInfo {
    uint32_t c0Count{ 0 };
    uint32_t repeat{ 0 };
    uint32_t repeatRounding{ 0 };
    uint32_t repeatRemaining{ 0 };
    uint32_t tail{ 0 };
};

enum class DataFormat : uint8_t {
    ND = 0,
    NZ,
    NCHW,
    NC1HWC0,
    NHWC,
};

struct PadParams {
    __aicore__ PadParams()
    {
        leftPad = 0;
        rightPad = 0;
        padValue = 0;
    }

    __aicore__ PadParams(const uint16_t leftPadIn, const uint16_t rightPadIn, const int32_t padValueIn)
    {
        leftPad = leftPadIn;
        rightPad = rightPadIn;
        padValue = padValueIn;
    }

    uint16_t leftPad = 0;
    uint16_t rightPad = 0;
    int32_t padValue = 0;
};

struct UnPadParams {
    __aicore__ UnPadParams()
    {
        leftPad = 0;
        rightPad = 0;
    }

    __aicore__ UnPadParams(const uint16_t leftPadIn, const uint16_t rightPadIn)
    {
        leftPad = leftPadIn;
        rightPad = rightPadIn;
    }

    uint16_t leftPad = 0;
    uint16_t rightPad = 0;
};

#if (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 200) || (__CCE_AICORE__ == 300)
// aipp config offset
constexpr int32_t AIPP_OFFSET_CSC_ENABLE = 63;
constexpr int32_t AIPP_OFFSET_CH1 = 16;
constexpr int32_t AIPP_OFFSET_CH2 = 32;
constexpr int32_t AIPP_OFFSET_CH3 = 48;
constexpr int32_t AIPP_OFFSET_SWAP_RB = 16;
constexpr int32_t AIPP_OFFSET_SWAP_UV = 17;
constexpr int32_t AIPP_OFFSET_SWAP_AX = 18;
constexpr int32_t AIPP_OFFSET_FORMAT = 19;
constexpr int32_t AIPP_OFFSET_SINGLE_LINE = 24;
constexpr int32_t AIPP_OFFSET_PADDING_MODE = 27;
constexpr int32_t AIPP_OFFSET_CPADDING_MODE = 40;
constexpr int32_t AIPP_OFFSET_CSC_OUT_CH0 = 16;
constexpr int32_t AIPP_OFFSET_CSC_OUT_CH1 = 24;
constexpr int32_t AIPP_OFFSET_CSC_OUT_CH2 = 32;
constexpr int32_t AIPP_OFFSET_CSC_IN_CH0 = 40;
constexpr int32_t AIPP_OFFSET_CSC_IN_CH1 = 48;
constexpr int32_t AIPP_OFFSET_CSC_IN_CH2 = 56;
constexpr int32_t AIPP_OFFSET_DTC_CH1 = 32;
constexpr int32_t AIPP_OFFSET_DTC_ROUND_MODE = 34;
#endif // (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 200) || (__CCE_AICORE__ == 300)

} // namespace AscendC
#endif // ASCENDC_MODULE_UTILS_STRUCT_CONFUSION_PAD_H