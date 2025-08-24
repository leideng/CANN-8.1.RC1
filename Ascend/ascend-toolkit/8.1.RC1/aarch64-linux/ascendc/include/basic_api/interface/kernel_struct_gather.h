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
 * \file kernel_struct_gather.h
 * \brief
 */
#ifndef ASCENDC_MODULE_STRUCT_GATHER_H
#define ASCENDC_MODULE_STRUCT_GATHER_H
#include "utils/kernel_utils_constants.h"

namespace AscendC {
struct GatherRepeatParams {
    __aicore__ GatherRepeatParams() {}

    __aicore__ GatherRepeatParams(const uint8_t dstBlkStrideIn, const uint8_t dstRepStrideIn)
        : dstBlkStride(dstBlkStrideIn),
          dstRepStride(dstRepStrideIn)
    {}

    uint32_t blockNumber = DEFAULT_BLK_NUM;
    uint16_t dstRepStride = DEFAULT_REPEAT_STRIDE;
    uint8_t dstBlkStride = DEFAULT_BLK_STRIDE;
    uint8_t src0BlkStride = DEFAULT_BLK_STRIDE;
    uint8_t src1BlkStride = DEFAULT_BLK_STRIDE;
    uint8_t src0RepStride = DEFAULT_REPEAT_STRIDE;
    uint8_t src1RepStride = DEFAULT_REPEAT_STRIDE;
    bool repeatStrideMode = false;
    bool strideSizeMode = false;
};

enum class GatherMaskMode : uint8_t {
    VERSION_V1 = 0,
    VERSION_V2 = 1
};

#if (__CCE_AICORE__ == 220)
const GatherMaskMode defaultGahterMaskMode = GatherMaskMode::VERSION_V2;
const GatherMaskMode defaultGatherMaskMode = GatherMaskMode::VERSION_V2;
#else
const GatherMaskMode defaultGahterMaskMode = GatherMaskMode::VERSION_V1;
const GatherMaskMode defaultGatherMaskMode = GatherMaskMode::VERSION_V1;
#endif

struct GatherMaskParams {
    __aicore__ GatherMaskParams() {}

    __aicore__ GatherMaskParams(const uint8_t src0BlockStrideIn, const uint16_t repeatTimesIn,
        const uint16_t src0RepeatStrideIn, const uint8_t src1RepeatStrideIn)
        : src0BlockStride(src0BlockStrideIn),
          repeatTimes(repeatTimesIn),
          src0RepeatStride(src0RepeatStrideIn),
          src1RepeatStride(src1RepeatStrideIn)
    {}

    uint8_t src0BlockStride = DEFAULT_BLK_STRIDE;
    uint16_t repeatTimes = 0;
    uint16_t src0RepeatStride = DEFAULT_REPEAT_STRIDE;
    uint8_t src1RepeatStride = DEFAULT_REPEAT_STRIDE;
};
} // namespace AscendC
#endif // ASCENDC_MODULE_STRUCT_GATHER_H