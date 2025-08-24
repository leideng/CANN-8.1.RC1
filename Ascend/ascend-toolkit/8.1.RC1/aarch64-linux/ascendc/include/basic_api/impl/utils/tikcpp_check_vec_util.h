/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file tikcpp_check_vec_util.h
 * \brief
 */
#ifndef ASCENDC_CHECK_VEC_UTIL_H
#define ASCENDC_CHECK_VEC_UTIL_H
#if ASCENDC_CPU_DEBUG
#include <string>
#include "kernel_utils.h"
#include "kernel_struct_transpose.h"
namespace AscendC {
namespace check {
struct VecScatterApiParams {
    VecScatterApiParams() {}
    VecScatterApiParams(uint64_t dstAddrIn, uint64_t srcAddrIn, uint64_t dstOffsetAddrIn, uint32_t dstDtypeBytesIn,
        uint32_t srcDtypeBytesIn, uint32_t dstOffsetDtypeBytesIn, uint32_t dstBaseAddrIn, uint32_t countIn,
        uint64_t dstSizeIn, uint64_t srcSizeIn, uint64_t dstOffsetSizeIn, uint8_t dstPosIn, uint8_t srcPosIn,
        uint8_t dstOffsetPosIn)
    {
        dstAddr = dstAddrIn;
        srcAddr = srcAddrIn;
        dstOffsetAddr = dstOffsetAddrIn;
        dstDtypeBytes = dstDtypeBytesIn;
        srcDtypeBytes = srcDtypeBytesIn;
        dstOffsetDtypeBytes = dstOffsetDtypeBytesIn;
        dstBaseAddr = dstBaseAddrIn;
        count = countIn;
        dstSize = dstSizeIn;
        srcSize = srcSizeIn;
        dstOffsetSize = dstOffsetSizeIn;
        dstLogicPos = dstPosIn;
        srcLogicPos = srcPosIn;
        dstOffsetLogicPos = dstOffsetPosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        srcPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(srcPosIn)));
        dstOffsetPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstOffsetPosIn)));
    }
    VecScatterApiParams(uint64_t dstAddrIn, uint64_t srcAddrIn, uint64_t dstOffsetAddrIn, uint32_t dstDtypeBytesIn,
        uint32_t srcDtypeBytesIn, uint32_t dstOffsetDtypeBytesIn, uint32_t dstBaseAddrIn, uint8_t repeatTimesIn,
        uint16_t srcRepStrideIn, uint64_t dstSizeIn, uint64_t srcSizeIn, uint64_t dstOffsetSizeIn, uint8_t dstPosIn,
        uint8_t srcPosIn, uint8_t dstOffsetPosIn)
    {
        dstAddr = dstAddrIn;
        srcAddr = srcAddrIn;
        dstOffsetAddr = dstOffsetAddrIn;
        dstDtypeBytes = dstDtypeBytesIn;
        srcDtypeBytes = srcDtypeBytesIn;
        dstOffsetDtypeBytes = dstOffsetDtypeBytesIn;
        dstBaseAddr = dstBaseAddrIn;
        repeatTimes = repeatTimesIn;
        srcRepStride = srcRepStrideIn;
        dstSize = dstSizeIn;
        srcSize = srcSizeIn;
        dstOffsetSize = dstOffsetSizeIn;
        dstLogicPos = dstPosIn;
        srcLogicPos = srcPosIn;
        dstOffsetLogicPos = dstOffsetPosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        srcPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(srcPosIn)));
        dstOffsetPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstOffsetPosIn)));
    }
    uint64_t dstAddr = 0;
    uint64_t srcAddr = 0;
    uint64_t dstOffsetAddr = 0;
    uint8_t repeatTimes = 0;
    uint32_t dstDtypeBytes = 0;
    uint32_t srcDtypeBytes = 0;
    uint32_t dstOffsetDtypeBytes = 0;
    uint32_t dstBaseAddr = 0;
    uint64_t dstSize = 0;
    uint64_t srcSize = 0;
    uint64_t dstOffsetSize = 0;
    uint16_t srcRepStride = 0;
    uint8_t dstLogicPos = 0;
    uint8_t srcLogicPos = 0;
    uint8_t dstOffsetLogicPos = 0;
    uint8_t dstPos = 0;
    uint8_t srcPos = 0;
    uint8_t dstOffsetPos = 0;
    uint32_t count = 0;
};
struct VecCmpRgtApiParams {
    VecCmpRgtApiParams() {}
    VecCmpRgtApiParams(uint64_t src0AddrIn, uint64_t src1AddrIn, uint16_t src0BlockStrideIn, uint16_t src1BlockStrideIn,
        uint16_t src0RepeatStrideIn, uint16_t src1RepeatStrideIn,  uint32_t src0DtypeBytesIn, uint32_t src1DtypeBytesIn,
        uint64_t src0SizeIn, uint64_t src1SizeIn, uint8_t src0PosIn, uint8_t src1PosIn)
    {
        src0Addr = src0AddrIn;
        src1Addr = src1AddrIn;
        src0BlockStride = src0BlockStrideIn;
        src1BlockStride = src1BlockStrideIn;
        src0RepeatStride = src0RepeatStrideIn;
        src1RepeatStride = src1RepeatStrideIn;
        src0DtypeBytes = src0DtypeBytesIn;
        src1DtypeBytes = src1DtypeBytesIn;
        src0Size = src0SizeIn;
        src1Size = src1SizeIn;
        src0LogicPos = src0PosIn;
        src1LogicPos = src1PosIn;
        src0Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src0PosIn)));
        src1Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src1PosIn)));
    }
    uint64_t src0Addr = 0;
    uint64_t src1Addr = 0;
    uint8_t repeatTimes = 1;
    uint16_t src0BlockStride = 0;
    uint16_t src1BlockStride = 0;
    uint16_t src0RepeatStride = 0;
    uint16_t src1RepeatStride = 0;
    uint32_t src0DtypeBytes = 0;
    uint32_t src1DtypeBytes = 0;
    uint64_t src0Size = 0;
    uint64_t src1Size = 0;
    uint8_t src0LogicPos = 0;
    uint8_t src1LogicPos = 0;
    uint8_t src0Pos = 0;
    uint8_t src1Pos = 0;
};
struct VectorPaddingApiParams {
    VectorPaddingApiParams() {}
    VectorPaddingApiParams(uint64_t dstAddrIn, uint64_t srcAddrIn, uint8_t repeatIn, uint16_t dstBlockStrideIn,
        uint16_t srcBlockStrideIn, uint16_t dstRepeatStrideIn, uint16_t srcRepeatStrideIn, uint32_t dstDtypeBytesIn,
        uint32_t srcDtypeBytesIn, uint64_t dstSizeIn, uint64_t srcSizeIn, uint8_t dstPosIn, uint8_t srcPosIn,
        uint8_t padModeIn, bool padSideIn)
    {
        dstAddr = dstAddrIn;
        srcAddr = srcAddrIn;
        repeatTimes = repeatIn;
        dstBlockStride = dstBlockStrideIn;
        srcBlockStride = srcBlockStrideIn;
        dstRepeatStride = dstRepeatStrideIn;
        srcRepeatStride = srcRepeatStrideIn;
        dstDtypeBytes = dstDtypeBytesIn;
        srcDtypeBytes = srcDtypeBytesIn;
        dstSize = dstSizeIn;
        srcSize = srcSizeIn;
        dstLogicPos = dstPosIn;
        srcLogicPos = srcPosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        srcPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(srcPosIn)));
        padMode = padModeIn;
        padSide = padSideIn;
    }
    VectorPaddingApiParams(uint64_t dstAddrIn, uint64_t srcAddrIn, uint32_t dstDtypeBytesIn, uint32_t srcDtypeBytesIn,
        uint64_t dstSizeIn, uint64_t srcSizeIn, uint8_t dstPosIn, uint8_t srcPosIn, uint32_t count, uint8_t padModeIn,
        bool padSideIn)
    {
        dstAddr = dstAddrIn;
        srcAddr = srcAddrIn;
        dstDtypeBytes = dstDtypeBytesIn;
        srcDtypeBytes = srcDtypeBytesIn;
        dstSize = dstSizeIn;
        srcSize = srcSizeIn;
        dstLogicPos = dstPosIn;
        srcLogicPos = srcPosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        srcPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(srcPosIn)));
        calCount = count;
        padMode = padModeIn;
        padSide = padSideIn;
    }
    uint64_t dstAddr = 0;
    uint64_t srcAddr = 0;
    uint8_t repeatTimes = 0;
    uint16_t dstBlockStride = 0;
    uint16_t srcBlockStride = 0;
    uint16_t dstRepeatStride = 0;
    uint16_t srcRepeatStride = 0;
    uint32_t dstDtypeBytes = 0;
    uint32_t srcDtypeBytes = 0;
    uint64_t dstSize = 0;
    uint64_t srcSize = 0;
    uint8_t dstLogicPos = 0;
    uint8_t srcLogicPos = 0;
    uint8_t dstPos = 0;
    uint8_t srcPos = 0;
    uint32_t calCount = 0;
    uint8_t padMode = 0;
    bool padSide = false;
};

struct VecBroadCastToMMApiParams {
    VecBroadCastToMMApiParams() {}
    VecBroadCastToMMApiParams(uint64_t dstAddrIn, uint64_t srcAddrIn, uint32_t dstDtypeBytesIn,
        uint32_t srcDtypeBytesIn, uint64_t dstSizeIn, uint64_t srcSizeIn, uint8_t dstPosIn, uint8_t srcPosIn,
        uint32_t blockCountIn, uint8_t blockLenIn, uint8_t srcGapIn, uint8_t dstGapIn)
    {
        dstAddr = dstAddrIn;
        srcAddr = srcAddrIn;
        dstDtypeBytes = dstDtypeBytesIn;
        srcDtypeBytes = srcDtypeBytesIn;
        dstSize = dstSizeIn;
        srcSize = srcSizeIn;
        dstLogicPos = dstPosIn;
        srcLogicPos = srcPosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        srcPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(srcPosIn)));
        blockCount = blockCountIn;
        blockLen = blockLenIn;
        srcGap = srcGapIn;
        dstGap = dstGapIn;
    }
    uint64_t dstAddr = 0;
    uint64_t srcAddr = 0;
    uint32_t dstDtypeBytes = 0;
    uint32_t srcDtypeBytes = 0;
    uint64_t dstSize = 0;
    uint64_t srcSize = 0;
    uint8_t dstLogicPos = 0;
    uint8_t srcLogicPos = 0;
    uint8_t dstPos = 0;
    uint8_t srcPos = 0;
    uint32_t calCount = 0;
    uint32_t blockCount = 0;
    uint8_t blockLen = 0;
    uint8_t srcGap = 0;
    uint8_t dstGap = 0;
};

struct VecBilinearInterpolationApiParams {
    VecBilinearInterpolationApiParams() {}
    VecBilinearInterpolationApiParams(uint64_t dstAddrIn, uint64_t src0AddrIn, uint64_t offsetAddrIn,
        uint64_t src1AddrIn, uint8_t hRepeatIn, bool repeatModeIn, uint16_t dstBlockStrideIn, uint16_t vROffsetIn,
        uint16_t vRepeatIn, uint32_t dstDtypeBytesIn, uint32_t src0DtypeBytesIn, uint32_t offsetDtypeBytesIn,
        uint32_t src1DtypeBytesIn, uint64_t dstSizeIn, uint64_t src0SizeIn, uint64_t offsetSizeIn, uint64_t src1SizeIn,
        uint8_t dstPosIn, uint8_t src0PosIn, uint8_t offsetPosIn, uint8_t src1PosIn)
    {
        dstAddr = dstAddrIn;
        src0Addr = src0AddrIn;
        offsetAddr = offsetAddrIn;
        src1Addr = src1AddrIn;
        hRepeat = hRepeatIn;
        repeatMode = repeatModeIn;
        dstBlockStride = dstBlockStrideIn;
        vROffset = vROffsetIn;
        vRepeat = vRepeatIn;
        dstDtypeBytes = dstDtypeBytesIn;
        src0DtypeBytes = src0DtypeBytesIn;
        offsetDtypeBytes = offsetDtypeBytesIn;
        src1DtypeBytes = src1DtypeBytesIn;
        dstSize = dstSizeIn;
        src0Size = src0SizeIn;
        offsetSize = offsetSizeIn;
        src1Size = src1SizeIn;
        dstLogicPos = dstPosIn;
        src0LogicPos = src0PosIn;
        offsetLogicPos = offsetPosIn;
        src1LogicPos = src1PosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));;
        src0Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src0PosIn)));;
        offsetPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(offsetPosIn)));;
        src1Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src1PosIn)));;
    }
    uint64_t dstAddr = 0;
    uint64_t src0Addr = 0;
    uint64_t offsetAddr = 0;
    uint64_t src1Addr = 0;
    uint8_t hRepeat = 0;
    bool repeatMode = false;
    uint16_t dstBlockStride = 0;
    uint16_t vROffset = 0;
    uint16_t vRepeat = 0;
    uint32_t dstDtypeBytes = 0;
    uint32_t src0DtypeBytes = 0;
    uint32_t offsetDtypeBytes = 0;
    uint32_t src1DtypeBytes = 0;
    uint64_t dstSize = 0;
    uint64_t src0Size = 0;
    uint64_t offsetSize = 0;
    uint64_t src1Size = 0;
    uint8_t dstLogicPos = 0;
    uint8_t src0LogicPos = 0;
    uint8_t offsetLogicPos = 0;
    uint8_t src1LogicPos = 0;
    uint8_t dstPos = 0;
    uint8_t src0Pos = 0;
    uint8_t offsetPos = 0;
    uint8_t src1Pos = 0;
};

struct VecTransposeApiParams {
    VecTransposeApiParams() {}
    VecTransposeApiParams(uint64_t dstAddrIn, uint64_t srcAddrIn, uint8_t repeatIn, uint16_t dstRepeatStrideIn,
        uint16_t srcRepeatStrideIn, uint32_t dstDtypeBytesIn, uint32_t srcDtypeBytesIn, uint64_t dstSizeIn,
        uint64_t srcSizeIn, uint8_t dstPosIn, uint8_t srcPosIn, int8_t indexIn = -1)
    {
        dstAddr = dstAddrIn;
        srcAddr = srcAddrIn;
        repeatTimes = repeatIn;
        dstRepeatStride = dstRepeatStrideIn;
        srcRepeatStride = srcRepeatStrideIn;
        dstDtypeBytes = dstDtypeBytesIn;
        srcDtypeBytes = srcDtypeBytesIn;
        dstSize = dstSizeIn;
        srcSize = srcSizeIn;
        dstLogicPos = dstPosIn;
        srcLogicPos = srcPosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        srcPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(srcPosIn)));
        index = indexIn;
    }
    VecTransposeApiParams(uint64_t dstAddrIn, uint64_t srcAddrIn, uint8_t repeatIn, uint16_t dstRepeatStrideIn,
        uint16_t srcRepeatStrideIn, uint32_t dstDtypeBytesIn, uint32_t srcDtypeBytesIn, uint64_t dstSizeIn,
        uint64_t srcSizeIn, uint64_t tmpBufferSizeIn, uint8_t dstPosIn, uint8_t srcPosIn, uint16_t nSizeIn,
        uint16_t cSizeIn, uint16_t hSizeIn, uint16_t wSizeIn, TransposeType transposeTypeIn)
    {
        dstAddr = dstAddrIn;
        srcAddr = srcAddrIn;
        repeatTimes = repeatIn;
        dstRepeatStride = dstRepeatStrideIn;
        srcRepeatStride = srcRepeatStrideIn;
        dstDtypeBytes = dstDtypeBytesIn;
        srcDtypeBytes = srcDtypeBytesIn;
        dstSize = dstSizeIn;
        srcSize = srcSizeIn;
        tmpBufferSize = tmpBufferSizeIn;
        dstLogicPos = dstPosIn;
        srcLogicPos = srcPosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        srcPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(srcPosIn)));
        nSize = nSizeIn;
        cSize = cSizeIn;
        hSize = hSizeIn;
        wSize = wSizeIn;
        transposeType = transposeTypeIn;
    }
    uint64_t dstAddr = 0;
    uint64_t srcAddr = 0;
    uint8_t repeatTimes = 0;
    uint16_t dstBlockStride = 1;
    uint16_t dstRepeatStride = 0;
    uint16_t srcBlockStride = 1;
    uint16_t srcRepeatStride = 0;
    uint32_t dstDtypeBytes = 0;
    uint64_t dstSize = 0;
    uint32_t srcDtypeBytes = 0;
    uint64_t srcSize = 0;
    uint64_t tmpBufferSize = 0;
    uint16_t nSize = 0;
    uint16_t cSize = 0;
    uint16_t hSize = 0;
    uint16_t wSize = 0;
    uint8_t dstLogicPos = 0;
    uint8_t srcLogicPos = 0;
    uint8_t dstPos = 0;
    uint8_t srcPos = 0;
    int8_t index = 0;   // transdata [16]
    TransposeType transposeType = TransposeType::TRANSPOSE_TYPE_NONE;
};

struct VecProposalApiParams {
    VecProposalApiParams() {}
    VecProposalApiParams(uint64_t dstAddrIn, uint64_t src0AddrIn, uint8_t repeatIn, uint32_t dstDtypeBytesIn,
        uint32_t src0DtypeBytesIn, uint64_t dstSizeIn, uint64_t src0SizeIn, uint8_t dstPosIn, uint8_t src0PosIn)
    {
        dstAddr = dstAddrIn;
        src0Addr = src0AddrIn;
        repeatTimes = repeatIn;
        dstDtypeBytes = dstDtypeBytesIn;
        src0DtypeBytes = src0DtypeBytesIn;
        dstSize = dstSizeIn;
        src0Size = src0SizeIn;
        dstLogicPos = dstPosIn;
        src0LogicPos = src0PosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        src0Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src0PosIn)));
    }
    VecProposalApiParams(uint64_t dstAddrIn, uint64_t src0AddrIn, uint64_t src1AddrIn, uint8_t repeatIn,
        uint32_t dstDtypeBytesIn, uint32_t src0DtypeBytesIn, uint32_t src1DtypeBytesIn, uint64_t dstSizeIn,
        uint64_t src0SizeIn, uint64_t src1SizeIn, uint8_t dstPosIn, uint8_t src0PosIn, uint8_t src1PosIn)
    {
        dstAddr = dstAddrIn;
        src0Addr = src0AddrIn;
        src1Addr = src1AddrIn;
        repeatTimes = repeatIn;
        dstDtypeBytes = dstDtypeBytesIn;
        src0DtypeBytes = src0DtypeBytesIn;
        src1DtypeBytes = src1DtypeBytesIn;
        dstSize = dstSizeIn;
        src0Size = src0SizeIn;
        src1Size = src1SizeIn;
        dstLogicPos = dstPosIn;
        src0LogicPos = src0PosIn;
        src1LogicPos = src1PosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        src0Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src0PosIn)));
        src1Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src1PosIn)));
    }
    VecProposalApiParams(uint64_t dstAddrIn, uint64_t src0AddrIn, uint8_t repeatIn, uint32_t dstDtypeBytesIn,
        uint32_t src0DtypeBytesIn, uint64_t dstSizeIn, uint64_t src0SizeIn, uint8_t dstPosIn, uint8_t src0PosIn,
        uint16_t validBitIn, const uint16_t elementLenIn[4], uint8_t srcIndexIn, bool isExhaustedIn = false,
        bool isContinousIn = false)
    {
        dstAddr = dstAddrIn;
        src0Addr = src0AddrIn;
        repeatTimes = repeatIn;
        dstDtypeBytes = dstDtypeBytesIn;
        src0DtypeBytes = src0DtypeBytesIn;
        dstSize = dstSizeIn;
        src0Size = src0SizeIn;
        dstLogicPos = dstPosIn;
        src0LogicPos = src0PosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        src0Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src0PosIn)));
        validBit = validBitIn;
        for (size_t i = 0; i < eleSize; i++) {
            elementLengths[i] = elementLenIn[i];
        }
        srcIndex = srcIndexIn;
        isExhausted = isExhaustedIn;
        isContinuous = isContinousIn;
    }
    const uint32_t eleSize = 4;
    uint64_t dstAddr = 0;
    uint64_t src0Addr = 0;
    uint64_t src1Addr = 0;
    uint8_t repeatTimes = 0;
    uint32_t dstDtypeBytes = 0;
    uint32_t src0DtypeBytes = 0;
    uint32_t src1DtypeBytes = 0;
    uint64_t dstSize = 0;
    uint64_t src0Size = 0;
    uint64_t src1Size = 0;
    uint8_t dstLogicPos = 0;
    uint8_t src0LogicPos = 0;
    uint8_t src1LogicPos = 0;
    uint8_t dstPos = 0;
    uint8_t src0Pos = 0;
    uint8_t src1Pos = 0;
    uint16_t validBit = 0;
    uint16_t elementLengths[4];
    uint8_t srcIndex = 0;
    bool isExhausted = false;
    bool isContinuous = false;
};
struct VecBroadCastApiParams {
    VecBroadCastApiParams() {}
    VecBroadCastApiParams(uint64_t dstAddrIn, uint64_t srcAddrIn, uint8_t repeatIn, uint16_t dstBlockStrideIn,
        uint16_t dstRepeatStrideIn, uint32_t dstDtypeBytesIn, uint32_t srcDtypeBytesIn, uint64_t dstSizeIn,
        uint64_t srcSizeIn, uint8_t dstPosIn, uint8_t srcPosIn)
    {
        dstAddr = dstAddrIn;
        srcAddr = srcAddrIn;
        repeatTimes = repeatIn;
        dstBlockStride = dstBlockStrideIn;
        dstRepeatStride = dstRepeatStrideIn;
        dstDtypeBytes = dstDtypeBytesIn;
        srcDtypeBytes = srcDtypeBytesIn;
        dstSize = dstSizeIn;
        srcSize = srcSizeIn;
        dstLogicPos = dstPosIn;
        srcLogicPos = srcPosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        srcPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(srcPosIn)));
    }
    uint64_t dstAddr = 0;
    uint64_t srcAddr = 0;
    uint8_t repeatTimes = 0;
    uint16_t dstBlockStride = 0;
    uint16_t dstRepeatStride = 0;
    uint32_t dstDtypeBytes = 0;
    uint32_t srcDtypeBytes = 0;
    uint64_t dstSize = 0;
    uint64_t srcSize = 0;
    uint8_t dstLogicPos = 0;
    uint8_t srcLogicPos = 0;
    uint8_t dstPos = 0;
    uint8_t srcPos = 0;
};
struct SortApiParams {
    SortApiParams() {}
    SortApiParams(uint64_t dstAddrIn, uint64_t concatAddrIn, uint64_t indexAddrIn, uint64_t tmpAddrIn, uint8_t repeatIn,
        uint32_t dstDtypeBytesIn, uint32_t concatDtypeBytesIn, uint32_t indexDtypeBytesIn, uint32_t tmpDtypeBytesIn,
        uint64_t dstSizeIn, uint64_t concatSizeIn, uint64_t indexSizeIn, uint64_t tmpSizeIn, uint8_t dstPosIn,
        uint8_t concatPosIn, uint8_t indexPosIn, uint8_t tmpPosIn, bool isFullSortIn)
    {
        dstAddr = dstAddrIn;
        concatAddr = concatAddrIn;
        indexAddr = indexAddrIn;
        tmpAddr = tmpAddrIn;
        repeatTimes = repeatIn;
        dstDtypeBytes = dstDtypeBytesIn;
        concatDtypeBytes = concatDtypeBytesIn;
        indexDtypeBytes = indexDtypeBytesIn;
        tmpDtypeBytes = tmpDtypeBytesIn;
        dstSize = dstSizeIn;
        concatSize = concatSizeIn;
        indexSize = indexSizeIn;
        tmpSize = tmpSizeIn;
        dstLogicPos = dstPosIn;
        concatLogicPos = concatPosIn;
        indexLogicPos = indexPosIn;
        tmpLogicPos = tmpPosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        concatPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(concatPosIn)));
        indexPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(indexPosIn)));
        tmpPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(tmpPosIn)));
        isFullSort = isFullSortIn;
    }
    uint64_t dstAddr = 0;
    uint64_t concatAddr = 0;
    uint64_t indexAddr = 0;
    uint64_t tmpAddr = 0;
    uint8_t repeatTimes = 0;
    uint32_t dstDtypeBytes = 0;
    uint32_t concatDtypeBytes = 0;
    uint32_t indexDtypeBytes = 0;
    uint32_t tmpDtypeBytes = 0;
    uint64_t dstSize = 0;
    uint64_t concatSize = 0;
    uint64_t indexSize = 0;
    uint64_t tmpSize = 0;
    uint8_t dstLogicPos = 0;
    uint8_t concatLogicPos = 0;
    uint8_t indexLogicPos = 0;
    uint8_t tmpLogicPos = 0;
    uint8_t dstPos = 0;
    uint8_t concatPos = 0;
    uint8_t indexPos = 0;
    uint8_t tmpPos = 0;
    bool isFullSort = false;
};
bool CheckFuncVecCmpRgtImplForMaskArray(VecCmpRgtApiParams& chkParams, const uint64_t mask[], const char* intriName);
bool CheckFuncVecCmpRgtImpl(VecCmpRgtApiParams& chkParams, const uint64_t mask, const char* intriName);
bool CheckVectorPaddingForMaskArray(VectorPaddingApiParams& chkParams, const uint64_t mask[], const char* intriName);
bool CheckVectorPadding(VectorPaddingApiParams& chkParams, const uint64_t mask, const char* intriName);
bool CheckVectorPadding(VectorPaddingApiParams& chkParams, const char* intriName);
bool CheckFuncBilinearInterpolationImpl(VecBilinearInterpolationApiParams& chkParams, const uint64_t mask,
    const char* intriName);
bool CheckFuncBilinearInterpolationImpl(VecBilinearInterpolationApiParams& chkParams, const uint64_t mask[],
    const char* intriName);
bool CheckFunTransposeImpl(VecTransposeApiParams& chkParams, const char* intriName);
bool CheckFunProposalImpl(VecProposalApiParams& chkParams, const char* intriName);
bool CheckFuncBroadCastToMMImpl(VecBroadCastToMMApiParams& chkParams, const char* intriName);
bool CheckFunScatterImpl(VecScatterApiParams& chkParams, const char* intriName);
bool CheckFunScatterImpl(VecScatterApiParams& chkParams, const uint64_t mask, const char* intriName);
bool CheckFunScatterImplForMaskArray(VecScatterApiParams& chkParams, const uint64_t mask[], const char* intriName);
bool CheckFunBcBImpl(VecBroadCastApiParams& chkParams, uint32_t dtypeSize, const char* intriName);
bool CheckSortImpl(SortApiParams& chkParams, const char* intriName);
} // namespace check
} // namespace AscendC
#endif
#endif