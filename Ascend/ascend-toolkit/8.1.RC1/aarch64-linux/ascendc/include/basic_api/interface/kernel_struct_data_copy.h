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
 * \file kernel_struct_data_copy.h
 * \brief
 */
#ifndef ASCENDC_MODULE_STRUCT_DATA_COPY_H
#define ASCENDC_MODULE_STRUCT_DATA_COPY_H
#include "utils/kernel_utils_mode.h"

namespace AscendC {
struct DataCopyParams {
    __aicore__ DataCopyParams() {}

    __aicore__ DataCopyParams(const uint16_t count, const uint16_t len, const uint16_t srcStrideIn,
        const uint16_t dstStrideIn)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn)
    {}

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint16_t blockLen = 0;
    uint16_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint16_t dstStride = DEFAULT_DATA_COPY_STRIDE;
};

struct DataCopyEnhancedParams {
    __aicore__ DataCopyEnhancedParams() {}

    __aicore__ DataCopyEnhancedParams(const BlockMode blockModeIn, const DeqScale deqScaleIn, const uint64_t deqValueIn,
        const uint8_t sidStoreModeIn, const bool isReluIn, const pad_t padModeIn, const uint64_t padValueIn)
        : blockMode(blockModeIn),
          deqScale(deqScaleIn),
          deqValue(deqValueIn),
          sidStoreMode(sidStoreModeIn),
          isRelu(isReluIn),
          padMode(padModeIn),
          padValue(padValueIn)
    {}

    BlockMode blockMode = BlockMode::BLOCK_MODE_NORMAL;
    DeqScale deqScale = DeqScale::DEQ_NONE;
    uint64_t deqValue = 0;
    uint8_t sidStoreMode = 0;
    bool isRelu = false;
    pad_t padMode = pad_t::PAD_NONE;
    uint64_t padValue = 0;
    uint64_t deqTensorAddr = 0;
};

struct DataCopyCO12DstParams {
    __aicore__ DataCopyCO12DstParams() {}

    __aicore__ DataCopyCO12DstParams(const uint16_t nSizeIn, const uint16_t mSizeIn, const uint32_t dstStrideIn,
        const uint16_t srcStrideIn, const QuantMode_t quantPreIn, const uint8_t reluPreIn, const bool channelSplitIn,
        const bool nz2ndEnIn)
        : nSize(nSizeIn),
          mSize(mSizeIn),
          dstStride(dstStrideIn),
          srcStride(srcStrideIn),
          quantPre(quantPreIn),
          reluPre(reluPreIn),
          channelSplit(channelSplitIn),
          nz2ndEn(nz2ndEnIn)
    {}

    uint8_t sid = 0;
    uint16_t nSize = 0;
    uint16_t mSize = 0;
    uint32_t dstStride = DEFAULT_DATA_COPY_STRIDE;
    uint16_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint8_t unitFlag = 0;
    uint8_t clipReluPre = 0;
    uint8_t eltWiseOp = 0;
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    uint8_t reluPre = 0;
    bool channelSplit = false;
    bool nz2ndEn = false;
};

struct DataCopyPadParams {
    __aicore__ DataCopyPadParams() {}

    __aicore__ DataCopyPadParams(const bool isPadValue, const uint8_t leftPadValue, const uint8_t rightPadValue,
        const uint64_t padValue)
        : isPad(isPadValue),
          leftPadding(leftPadValue),
          rightPadding(rightPadValue),
          paddingValue(padValue)
    {}

    bool isPad = false;
    uint8_t leftPadding = 0;
    uint8_t rightPadding = 0;
    uint64_t paddingValue = 0;
};

struct DataCopyExtParams {
    __aicore__ DataCopyExtParams() {}

    __aicore__ DataCopyExtParams(const uint16_t count, const uint32_t len, const uint32_t srcStrideIn,
        const uint32_t dstStrideIn, const uint32_t rsvIn)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          rsv(rsvIn)
    {}

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint32_t blockLen = 0;
    uint32_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint32_t dstStride = DEFAULT_DATA_COPY_STRIDE;
    uint32_t rsv = 0; // reserved information
};

template <typename T>
struct DataCopyPadExtParams {
    __aicore__ DataCopyPadExtParams() {}

    __aicore__ DataCopyPadExtParams(const bool isPadValue, const uint8_t leftPadValue, const uint8_t rightPadValue,
        T padValue)
        : isPad(isPadValue),
          leftPadding(leftPadValue),
          rightPadding(rightPadValue),
          paddingValue(padValue)
    {}

    bool isPad = false;
    uint8_t leftPadding = 0;
    uint8_t rightPadding = 0;
    T paddingValue = 0;
};

struct Nd2NzParams {
    __aicore__ Nd2NzParams() {}

    __aicore__ Nd2NzParams(const uint16_t ndNumIn, const uint16_t nValueIn, const uint16_t dValueIn,
        const uint16_t srcNdMatrixStrideIn, const uint16_t srcDValueIn, const uint16_t dstNzC0StrideIn,
        const uint16_t dstNzNStrideIn, const uint16_t dstNzMatrixStrideIn)
        : ndNum(ndNumIn),
          nValue(nValueIn),
          dValue(dValueIn),
          srcNdMatrixStride(srcNdMatrixStrideIn),
          srcDValue(srcDValueIn),
          dstNzC0Stride(dstNzC0StrideIn),
          dstNzNStride(dstNzNStrideIn),
          dstNzMatrixStride(dstNzMatrixStrideIn)
    {}

    uint16_t ndNum = 0;
    uint16_t nValue = 0;
    uint16_t dValue = 0;
    uint16_t srcNdMatrixStride = 0;
    uint16_t srcDValue = 0;
    uint16_t dstNzC0Stride = 0;
    uint16_t dstNzNStride = 0;
    uint16_t dstNzMatrixStride = 0;
};

struct Nz2NdParamsFull {
    __aicore__ Nz2NdParamsFull() {}

    __aicore__ Nz2NdParamsFull(const uint16_t ndNumIn, const uint16_t nValueIn, const uint16_t dValueIn,
        const uint16_t srcNdMatrixStrideIn, const uint16_t srcNStrideIn, const uint16_t dstDStrideIn,
        const uint16_t dstNdMatrixStrideIn)
        : ndNum(ndNumIn),
          nValue(nValueIn),
          dValue(dValueIn),
          srcNdMatrixStride(srcNdMatrixStrideIn),
          srcNStride(srcNStrideIn),
          dstDStride(dstDStrideIn),
          dstNdMatrixStride(dstNdMatrixStrideIn)
    {}

    uint16_t ndNum = 1;
    uint16_t nValue = 0;
    uint16_t dValue = 0;
    uint16_t srcNdMatrixStride = 1;
    uint16_t srcNStride = 0;
    uint16_t dstDStride = 0;
    uint16_t dstNdMatrixStride = 1;
};

struct SliceInfo {
    __aicore__ SliceInfo() {}

    __aicore__ SliceInfo(const uint32_t startIndexIn, const uint32_t endIndexIn, const uint32_t strideIn,
        const uint32_t burstLenIn, const uint32_t shapeValueIn = 0)
        : startIndex(startIndexIn),
          endIndex(endIndexIn),
          stride(strideIn),
          burstLen(burstLenIn),
          shapeValue(shapeValueIn)
    {}

    uint32_t startIndex = 0;
    uint32_t endIndex = ONE_BLK_SIZE - 1;
    uint32_t stride = 0;
    uint32_t burstLen = ONE_BLK_SIZE;
    uint32_t shapeValue = 0;
};

struct CopyRepeatParams {
    __aicore__ CopyRepeatParams() {}

    __aicore__ CopyRepeatParams(const uint16_t dstStrideIn, const uint16_t srcStrideIn, uint16_t dstRepeatSizeIn,
        uint16_t srcRepeatSizeIn)
        : dstStride(dstStrideIn),
          srcStride(srcStrideIn),
          dstRepeatSize(dstRepeatSizeIn),
          srcRepeatSize(srcRepeatSizeIn)
    {}

    uint16_t dstStride = DEFAULT_DATA_COPY_STRIDE;
    uint16_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint16_t dstRepeatSize = DEFAULT_REPEAT_STRIDE;
    uint16_t srcRepeatSize = DEFAULT_REPEAT_STRIDE;
};
} // namespace AscendC
#endif // ASCENDC_MODULE_STRUCT_DATA_COPY_H