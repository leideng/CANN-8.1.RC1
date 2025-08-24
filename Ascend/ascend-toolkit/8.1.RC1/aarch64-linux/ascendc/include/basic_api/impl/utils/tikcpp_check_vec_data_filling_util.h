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
 * \file tikcpp_check_vec_data_filling_util.h
 * \brief
 */
#ifndef ASCENDC_CHECK_VEC_DATA_FILLING_UTIL_H
#define ASCENDC_CHECK_VEC_DATA_FILLING_UTIL_H
#if ASCENDC_CPU_DEBUG
#include <string>
#include "kernel_utils.h"
namespace AscendC {
namespace check {
struct VecDupApiParams {
    VecDupApiParams() {}
    VecDupApiParams(uint64_t dstAddrIn, uint8_t repeatIn, uint16_t dstBlockStrideIn, uint16_t dstRepeatStrideIn,
        uint32_t dstDtypeBytesIn, uint64_t dstSizeIn, uint8_t dstPosIn)
    {
        dstAddr = dstAddrIn;
        repeatTimes = repeatIn;
        dstBlockStride = dstBlockStrideIn;
        dstRepeatStride = dstRepeatStrideIn;
        dstDtypeBytes = dstDtypeBytesIn;
        dstSize = dstSizeIn;
        dstLogicPos = dstPosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
    }
    VecDupApiParams(uint64_t dstAddrIn, uint32_t dstDtypeBytesIn, uint64_t dstSizeIn, uint8_t dstPosIn, uint32_t count)
    {
        dstAddr = dstAddrIn;
        dstDtypeBytes = dstDtypeBytesIn;
        dstSize = dstSizeIn;
        dstLogicPos = dstPosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        calCount = count;
    }
    uint64_t dstAddr = 0;
    uint8_t repeatTimes = 0;
    uint16_t dstBlockStride = 0;
    uint16_t dstRepeatStride = 0;
    uint32_t dstDtypeBytes = 0;
    uint64_t dstSize = 0;
    uint8_t dstLogicPos = 0;
    uint8_t dstPos = 0;
    uint32_t calCount = 0;
};
struct VecCreateVecIndexApiParams {
    VecCreateVecIndexApiParams() {}
    VecCreateVecIndexApiParams(uint64_t dstAddrIn, uint8_t repeatIn, uint16_t dstBlkStrideIn, uint16_t dstRepStrideIn,
        uint32_t dstDtypeBytesIn, uint64_t dstSizeIn, uint8_t dstPosIn, uint32_t calCountIn)
    {
        dstAddr = dstAddrIn;
        repeatTimes = repeatIn;
        dstBlkStride = dstBlkStrideIn;
        dstRepStride = dstRepStrideIn;
        dstDtypeBytes = dstDtypeBytesIn;
        dstSize = dstSizeIn;
        dstLogicPos = dstPosIn;
        dstPos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(dstPosIn)));
        calCount = calCountIn;
    }
    uint64_t dstAddr = 0;
    uint8_t repeatTimes = 0;
    uint16_t dstBlkStride = 0;
    uint16_t dstRepStride = 0;
    uint32_t dstDtypeBytes = 0;
    uint64_t dstSize = 0;
    uint8_t dstLogicPos = 0;
    uint8_t dstPos = 0;
    uint32_t calCount = 0;
};
bool CheckFunDupImplForMaskArray(VecDupApiParams& chkParams, const uint64_t mask[], const char* intriName);
bool CheckFunDupImpl(VecDupApiParams& chkParams, const uint64_t mask, const char* intriName);
bool CheckFunDupImpl(VecDupApiParams& chkParams, const char* intriName);
bool CheckFuncCreateVecIndexImpl(VecCreateVecIndexApiParams& chkParams, const uint64_t mask, const char* intriName);
bool CheckFuncCreateVecIndexImpl(VecCreateVecIndexApiParams& chkParams, const uint64_t mask[], const char* intriName);
bool CheckFuncCreateVecIndexImpl(VecCreateVecIndexApiParams& chkParams, const char* intriName);
} // namespace check
} // namespace AscendC
#endif
#endif