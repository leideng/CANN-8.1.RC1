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
 * \file kernel_utils_struct_param.h
 * \brief
 */
#ifndef ASCENDC_MODULE_UTILS_STRUCT_PARAM_H
#define ASCENDC_MODULE_UTILS_STRUCT_PARAM_H
#include "utils/kernel_utils_mode.h"

namespace AscendC {
struct ReduceRepeatParams {
    __aicore__ ReduceRepeatParams()
    {
        highMask = FULL_MASK;
        lowMask = FULL_MASK;
        repeatTimes = 0;
        dstRepStride = DEFAULT_REDUCE_DST_REP_SRIDE; // dst Stride Unit is 2B(fp16)/4B(fp32)
        srcBlkStride = DEFAULT_BLK_STRIDE;
        srcRepStride = DEFAULT_REPEAT_STRIDE; // src Stride Unit is 32B
    }

    __aicore__ ReduceRepeatParams(const int32_t mask, const int32_t repeatTimesIn, const int32_t dstRepStrideIn,
        const int32_t srcBlkStrideIn, const int32_t srcRepStrideIn)
    {
#if __CCE_AICORE__ == 300 || defined(__DAV_M310__)
        normalMask = mask;
        maskMode = 1;
#else
        if (mask == HLAF_MASK_LEN) {
            highMask = 0;
            lowMask = FULL_MASK;
        } else if (mask == HLAF_MASK_LEN * DOUBLE_FACTOR) {
            highMask = FULL_MASK;
            lowMask = FULL_MASK;
        } else {
            highMask = (mask > HLAF_MASK_LEN) ?
                (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(mask - HLAF_MASK_LEN)) - 1) :
                0;
            lowMask =
                (mask > HLAF_MASK_LEN) ? FULL_MASK : (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(mask)) - 1);
        }
#endif
        repeatTimes = repeatTimesIn;
        dstRepStride = dstRepStrideIn;
        srcBlkStride = srcBlkStrideIn;
        srcRepStride = srcRepStrideIn;
    }

    __aicore__ ReduceRepeatParams(const uint64_t mask[2], const int32_t repeatTimesIn, const int32_t dstRepStrideIn,
        const int32_t srcBlkStrideIn, const int32_t srcRepStrideIn)
    {
#if __CCE_AICORE__ == 300 || defined(__DAV_M310__)
        bitMask[0] = mask[0];
        bitMask[1] = mask[1];
#else
        highMask = mask[1];
        lowMask = mask[0];
#endif
        repeatTimes = repeatTimesIn;
        dstRepStride = dstRepStrideIn;
        srcBlkStride = srcBlkStrideIn;
        srcRepStride = srcRepStrideIn;
    }

    uint64_t highMask = 0;
    uint64_t lowMask = 0;
    uint64_t bitMask[2] = {0, 0};
    int32_t normalMask = 0;
    int32_t maskMode = 0;
    int32_t repeatTimes = 0;
    int32_t dstRepStride = 0;
    int32_t srcBlkStride = 0;
    int32_t srcRepStride = 0;
};

struct DumpMessageHead {
    __aicore__ DumpMessageHead()
    {
        type = 0;
        lenth = 0;
        addr = 0;
        dataType = 0;
        desc = 0;
        bufferId = 0;
        position = 0;
        rsv = 0;
    }

    __aicore__ DumpMessageHead(uint32_t typeIn, uint32_t lenthIn, uint32_t addrIn, uint32_t dataTypeIn, uint32_t descIn,
        uint32_t bufferIdIn, uint32_t positionIn, uint32_t rsvIn)
    {
        type = typeIn;
        lenth = lenthIn;
        addr = addrIn;
        dataType = dataTypeIn;
        desc = descIn;
        bufferId = bufferIdIn;
        position = positionIn;
        rsv = rsvIn;
    }

    uint32_t type = 0; // Dump Type 1:DumpScalar(DUMP_SCALAR), 2:DumpTensor (DUMP_TENSOR)
    uint32_t lenth = 0;
    uint32_t addr = 0;     // Dumptensor address, DumpScalar:0
    uint32_t dataType = 0; // data type: int32_t/half/...
    uint32_t desc = 0;     // for usr to add info or tag
    uint32_t bufferId = 0; // DumpScalar: Blockid, DumpTensor: UB adddr ()
    uint32_t position = 0; // DumpScalar: 0: MIX, 1: AIC 2: AIV; DumpTensor: 1:UB, 2:L1
    uint32_t rsv = 0;      // reserved information
};

struct DumpShapeMessageHead {
    __aicore__ DumpShapeMessageHead()
    {
        dim = 0;
        rsv = 0;
        for (uint32_t idx = 0; idx < K_MAX_SHAPE_DIM; ++idx) {
            shape[idx] = 0;
        }
    }

    __aicore__ DumpShapeMessageHead(uint32_t dimIn, uint32_t shapeIn[], uint32_t rsvIn = 0)
    {
        ASCENDC_ASSERT((dimIn <= K_MAX_SHAPE_DIM), {
            KERNEL_LOG(KERNEL_ERROR, "dim is %u, which should be less than %d", dimIn, K_MAX_SHAPE_DIM);
        });
        dim = dimIn;
        rsv = rsvIn;
        for (uint32_t idx = 0; idx < K_MAX_SHAPE_DIM; ++idx) {
            if (idx < dim) {
                shape[idx] = shapeIn[idx];
            } else {
                shape[idx] = 0;
            }
        }
    }

    uint32_t dim = 0;
    uint32_t shape[K_MAX_SHAPE_DIM];
    uint32_t rsv = 0;      // reserved information
};

struct ProposalIntriParams {
    __aicore__ ProposalIntriParams()
    {
        repeat = 0;
        modeNumber = 0;
    }

    __aicore__ ProposalIntriParams(const int32_t repeatTimes, const int32_t modeNumberIn)
    {
        repeat = repeatTimes;      // [1,255]
        modeNumber = modeNumberIn; // modeNumberIn: 0: x1, 1: y1, 2: x2, 3: y2, 4: score, 5:label
    }

    int32_t repeat = 0;
    int32_t modeNumber = 0;
};

struct BlockInfo {
    __aicore__ BlockInfo()
    {
        len = 0;
        core = 0;
        blockNum = 0;
        dumpOffset = 0;
        magic = 0;
        rsv = 0;
        dumpAddr = 0;
    }
    __aicore__ BlockInfo(uint64_t dumpAddrIn, uint32_t lenIn, uint32_t coreIn, uint32_t blockNumIn,
        uint32_t dumpOffsetIn, uint32_t magicIn, uint32_t rsvIn)
    {
        len = lenIn;
        core = coreIn;
        blockNum = blockNumIn;
        dumpOffset = dumpOffsetIn;
        magic = magicIn;
        rsv = rsvIn;
        dumpAddr = dumpAddrIn;
    }
    uint32_t len = 0;
    uint32_t core = 0;       // current core id
    uint32_t blockNum = 0;   // total core num
    uint32_t dumpOffset = 0; // size used by current core
    uint32_t magic = 0;      // magic number
    uint32_t rsv = 0;
    uint64_t dumpAddr = 0; // start addr of dump
};

struct DumpMeta {
    uint32_t typeId = static_cast<uint32_t>(DumpType::DUMP_META);
    uint32_t len = 8;
    uint16_t blockDim = 0;
    uint8_t coreType = 0;
    uint8_t taskRation = 0;
    uint32_t rsv = 0;
};

struct DumpTimeStamp {
    uint32_t typeId = static_cast<uint32_t>(DumpType::DUMP_TIME_STAMP);
    uint32_t len = 24;
    uint32_t descId = 0;
    uint32_t rsv = 0;
    uint64_t systemCycle = 0;
    uint64_t pcPtr = 0;
};
} // namespace AscendC
#endif // ASCENDC_MODULE_UTILS_STRUCT_PARAM_H