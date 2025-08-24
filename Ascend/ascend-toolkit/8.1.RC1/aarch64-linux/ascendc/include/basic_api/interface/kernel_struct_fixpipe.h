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
 * \file kernel_struct_fixpipe.h
 * \brief
 */
#ifndef ASCENDC_MODULE_STRUCT_FIXPIPE_H
#define ASCENDC_MODULE_STRUCT_FIXPIPE_H

namespace AscendC {
enum class CO2Layout : uint8_t {
    NZ = 0,
    ROW_MAJOR, // ND Row
    COLUMN_MAJOR // ND Column
};

struct FixpipeConfig {
    CO2Layout format;
};

constexpr FixpipeConfig CFG_NZ = {CO2Layout::NZ};
constexpr FixpipeConfig CFG_ROW_MAJOR = {CO2Layout::ROW_MAJOR};
constexpr FixpipeConfig CFG_COLUMN_MAJOR = {CO2Layout::COLUMN_MAJOR};

struct FixpipeParamsV220 {
    __aicore__ FixpipeParamsV220() {}

    __aicore__ FixpipeParamsV220(const uint16_t nSizeIn, const uint16_t mSizeIn, const uint16_t srcStrideIn,
        const uint32_t dstStrideIn, const bool reluEnIn)
        : nSize(nSizeIn),
          mSize(mSizeIn),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          reluEn(reluEnIn)
    {}

    __aicore__ FixpipeParamsV220(const uint16_t nSizeIn, const uint16_t mSizeIn, const uint16_t srcStrideIn,
        const uint32_t dstStrideIn, const bool reluEnIn, const QuantMode_t quantPreIn, const int64_t deqScalarIn,
        const uint16_t ndNumIn, const uint16_t srcNdStrideIn, const uint16_t dstNdStrideIn, const uint8_t unitFlagIn)
        : nSize(nSizeIn),
          mSize(mSizeIn),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          reluEn(reluEnIn),
          quantPre(quantPreIn),
          deqScalar(deqScalarIn),
          ndNum(ndNumIn),
          srcNdStride(srcNdStrideIn),
          dstNdStride(dstNdStrideIn),
          unitFlag(unitFlagIn)
    {}

    __aicore__ FixpipeParamsV220(const uint16_t nSizeIn, const uint16_t mSizeIn, const uint16_t srcStrideIn,
        const uint32_t dstStrideIn, const bool reluEnIn, const QuantMode_t quantPreIn, const int64_t deqScalarIn,
        const uint16_t ndNumIn, const uint16_t srcNdStrideIn, const uint16_t dstNdStrideIn, const uint8_t unitFlagIn,
        const bool isChannelSplitIn)
        : nSize(nSizeIn),
          mSize(mSizeIn),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          reluEn(reluEnIn),
          quantPre(quantPreIn),
          deqScalar(deqScalarIn),
          ndNum(ndNumIn),
          srcNdStride(srcNdStrideIn),
          dstNdStride(dstNdStrideIn),
          unitFlag(unitFlagIn),
          isChannelSplit(isChannelSplitIn)
    {}

    uint16_t nSize = 0;
    uint16_t mSize = 0;  // M-DirectionSize
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
    // Params: used for Quant
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    uint64_t deqScalar;
    // Params: used for nz2nd
    uint16_t ndNum = 1;
    uint16_t srcNdStride = 0;
    uint16_t dstNdStride = 0;
    bool reluEn = false;
    uint8_t unitFlag = 0;
    bool isChannelSplit = false;
};

using FixpipeParamsM300 = FixpipeParamsV220;
using FixpipeParamsM310 = FixpipeParamsV220;
} // namespace AscendC
#endif // ASCENDC_MODULE_STRUCT_FIXPIPE_H