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
 * \file tikcpp_check_vec_reduce_util.h
 * \brief
 */

#ifndef ASCENDC_CHECK_VEC_REDUCE_UTIL_H
#define ASCENDC_CHECK_VEC_REDUCE_UTIL_H
#if ASCENDC_CPU_DEBUG
#include <string>
#include "kernel_utils.h"
namespace AscendC {
namespace check {
struct VecReduceApiParams {
    VecReduceApiParams() {}
    VecReduceApiParams(uint64_t dstAddrIn, uint64_t src0AddrIn, uint32_t dstDtypeBytesIn, uint32_t src0DtypeBytesIn,
        int32_t repeatIn, uint16_t dstRepStrideIn, uint16_t srcBlkStrideIn, uint16_t srcRepStrideIn, uint64_t dstSizeIn,
        uint64_t src0SizeIn, uint8_t dstPosIn, uint8_t src0PosIn)
    {
        dstAddr = dstAddrIn;
        src0Addr = src0AddrIn;
        dstDtypeBytes = dstDtypeBytesIn;
        src0DtypeBytes = src0DtypeBytesIn;
        repeatTimes = repeatIn;
        dstRepeatStride = dstRepStrideIn;
        src0BlockStride = srcBlkStrideIn;
        src0RepeatStride = srcRepStrideIn;
        dstSize = dstSizeIn;
        src0Size = src0SizeIn;
        dstLogicPos = dstPosIn;
        src0LogicPos = src0PosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        src0Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src0PosIn)));
    }
    VecReduceApiParams(uint64_t dstAddrIn, uint64_t src0AddrIn, uint64_t src1AddrIn, uint32_t dstDtypeBytesIn,
        uint32_t src0DtypeBytesIn, uint32_t src1DtypeBytesIn, int32_t repeatIn, uint32_t calCountIn, bool calIndexIn,
        uint64_t dstSizeIn, uint64_t src0SizeIn, uint64_t src1SizeIn, uint8_t dstPosIn, uint8_t src0PosIn,
        uint8_t src1PosIn)
    {
        dstAddr = dstAddrIn;
        src0Addr = src0AddrIn;
        src1Addr = src1AddrIn;
        dstDtypeBytes = dstDtypeBytesIn;
        src0DtypeBytes = src0DtypeBytesIn;
        src1DtypeBytes = src1DtypeBytesIn;
        repeatTimes = repeatIn;
        calCount = calCountIn;
        calIndex = calIndexIn;
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

    VecReduceApiParams(uint64_t dstAddrIn, uint64_t src0AddrIn, uint64_t src1AddrIn, uint32_t dstDtypeBytesIn,
        uint32_t src0DtypeBytesIn, uint32_t src1DtypeBytesIn, int32_t repeatIn, uint32_t calCountIn, uint64_t dstSizeIn,
        uint64_t src0SizeIn, uint64_t src1SizeIn, uint8_t dstPosIn, uint8_t src0PosIn, uint8_t src1PosIn)
    {
        dstAddr = dstAddrIn;
        src0Addr = src0AddrIn;
        src1Addr = src1AddrIn;
        dstDtypeBytes = dstDtypeBytesIn;
        src0DtypeBytes = src0DtypeBytesIn;
        src1DtypeBytes = src1DtypeBytesIn;
        repeatTimes = repeatIn;
        calCount = calCountIn;
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

    VecReduceApiParams(uint64_t dstAddrIn, uint64_t src0AddrIn, uint64_t src1AddrIn, uint32_t dstDtypeBytesIn,
        uint32_t src0DtypeBytesIn, uint32_t src1DtypeBytesIn, int32_t repeatIn, bool calIndexIn, uint64_t dstSizeIn,
        uint64_t src0SizeIn, uint64_t src1SizeIn, uint8_t dstPosIn, uint8_t src0PosIn, uint8_t src1PosIn,
        uint16_t src0RepeatStrideIn)
    {
        dstAddr = dstAddrIn;
        src0Addr = src0AddrIn;
        src1Addr = src1AddrIn;
        dstDtypeBytes = dstDtypeBytesIn;
        src0DtypeBytes = src0DtypeBytesIn;
        src1DtypeBytes = src1DtypeBytesIn;
        repeatTimes = repeatIn;
        calIndex = calIndexIn;
        dstSize = dstSizeIn;
        src0Size = src0SizeIn;
        src1Size = src1SizeIn;
        dstLogicPos = dstPosIn;
        src0LogicPos = src0PosIn;
        src1LogicPos = src1PosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        src0Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src0PosIn)));
        src1Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src1PosIn)));
        src0RepeatStride = src0RepeatStrideIn;
    }

    VecReduceApiParams(uint64_t dstAddrIn, uint64_t src0AddrIn, uint64_t src1AddrIn, uint32_t dstDtypeBytesIn,
        uint32_t src0DtypeBytesIn, uint32_t src1DtypeBytesIn, int32_t repeatIn, uint64_t dstSizeIn, uint64_t src0SizeIn,
        uint64_t src1SizeIn, uint8_t dstPosIn, uint8_t src0PosIn, uint8_t src1PosIn, uint16_t src0RepeatStrideIn)
    {
        dstAddr = dstAddrIn;
        src0Addr = src0AddrIn;
        src1Addr = src1AddrIn;
        dstDtypeBytes = dstDtypeBytesIn;
        src0DtypeBytes = src0DtypeBytesIn;
        src1DtypeBytes = src1DtypeBytesIn;
        repeatTimes = repeatIn;
        dstSize = dstSizeIn;
        src0Size = src0SizeIn;
        src1Size = src1SizeIn;
        dstLogicPos = dstPosIn;
        src0LogicPos = src0PosIn;
        src1LogicPos = src1PosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        src0Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src0PosIn)));
        src1Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src1PosIn)));
        src0RepeatStride = src0RepeatStrideIn;
    }

    uint64_t dstAddr = 0;
    uint64_t src0Addr = 0;
    uint64_t src1Addr = 0;
    uint8_t repeatTimes = 0;
    uint16_t dstBlockStride = 0;
    uint16_t src0BlockStride = 0;
    uint16_t src1BlockStride = 0;
    uint16_t dstRepeatStride = 0;
    uint16_t src0RepeatStride = 0;
    uint16_t src1RepeatStride = 0;
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
    uint32_t calCount = 0;
    bool calIndex = 0;
};

struct VecReduceWhlApiParams {
    VecReduceWhlApiParams() {}
    VecReduceWhlApiParams(uint64_t dstAddrIn, uint64_t src0AddrIn, uint32_t dstDtypeBytesIn, uint32_t src0DtypeBytesIn,
        int32_t repeatIn, uint16_t dstRepStrideIn, uint16_t srcBlkStrideIn, uint16_t srcRepStrideIn,
        ReduceOrder orderIn, uint64_t dstSizeIn, uint64_t src0SizeIn, uint8_t dstPosIn, uint8_t src0PosIn)
    {
        dstAddr = dstAddrIn;
        src0Addr = src0AddrIn;
        dstDtypeBytes = dstDtypeBytesIn;
        src0DtypeBytes = src0DtypeBytesIn;
        repeatTimes = repeatIn;
        dstRepeatStride = dstRepStrideIn;
        src0BlockStride = srcBlkStrideIn;
        src0RepeatStride = srcRepStrideIn;
        order = orderIn;
        dstSize = dstSizeIn;
        src0Size = src0SizeIn;
        dstLogicPos = dstPosIn;
        src0LogicPos = src0PosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        src0Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(src0PosIn)));
    }

    uint64_t dstAddr = 0;
    uint64_t src0Addr = 0;
    uint8_t repeatTimes = 0;
    uint16_t src0BlockStride = 0;
    uint16_t dstRepeatStride = 0;
    uint16_t src0RepeatStride = 0;
    uint32_t dstDtypeBytes = 0;
    uint32_t src0DtypeBytes = 0;
    ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX;
    uint64_t dstSize = 0;
    uint64_t src0Size = 0;
    uint8_t dstLogicPos = 0;
    uint8_t src0LogicPos = 0;
    uint8_t dstPos = 0;
    uint8_t src0Pos = 0;
};

bool CheckFunReduceImplForMaskArray(VecReduceApiParams& chkParams, const uint64_t mask[], const char* intriName);
bool CheckFunReduceImpl(VecReduceApiParams& chkParams, const uint64_t mask, const char* intriName);
bool CheckFunReduceImpl(VecReduceApiParams& chkParams, const char* intriName);

bool CheckFunReduceOtherImplForMaskArray(VecReduceApiParams& chkParams, const uint64_t mask[], const char* intriName);
bool CheckFunReduceOtherImpl(VecReduceApiParams& chkParams, const uint64_t mask, const char* intriName);

bool CheckFunReduceOtherWhlImplForMaskArray(VecReduceWhlApiParams& chkParams, const uint64_t mask[], const char* intriName);
bool CheckFunReduceOtherWhlImpl(VecReduceWhlApiParams& chkParams, const uint64_t mask, const char* intriName);

} // namespace check
} // namespace AscendC
#endif
#endif