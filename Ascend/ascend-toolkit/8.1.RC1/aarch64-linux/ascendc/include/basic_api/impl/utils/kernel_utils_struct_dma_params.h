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
 * \file kernel_utils_struct_dma_params.h
 * \brief
 */
#ifndef ASCENDC_MODULE_UTILS_STRUCT_DMA_PARAMS_H
#define ASCENDC_MODULE_UTILS_STRUCT_DMA_PARAMS_H
#include "utils/kernel_utils_mode.h"

namespace AscendC {
struct QuantParams {
    __aicore__ QuantParams() {}
    __aicore__ QuantParams(const QuantMode_t quantPreIn) : quantPre(quantPreIn) {}
    __aicore__ QuantParams(const QuantMode_t quantPreIn, const uint64_t deqScalarIn)
        : quantPre(quantPreIn), deqScalar(deqScalarIn) {}
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    uint64_t deqScalar;
};

struct Nz2NdParams {
    __aicore__ Nz2NdParams()
    {
        nz2ndEn = false;
        ndNum = 1;
        srcNdStride = 0;
        dstNdStride = 0;
        originalNSize = 0;
    }

    __aicore__ Nz2NdParams(const bool nz2ndEnIn, const uint16_t ndNumIn, const uint16_t srcNdStrideIn,
        const uint16_t dstNdStrideIn, const uint16_t originalNSizeIn)
    {
        nz2ndEn = nz2ndEnIn;
        ndNum = ndNumIn;
        srcNdStride = srcNdStrideIn;
        dstNdStride = dstNdStrideIn;
        originalNSize = originalNSizeIn;
    }

    bool nz2ndEn = false;
    uint16_t ndNum = 1;
    uint16_t srcNdStride = 0;
    uint16_t dstNdStride = 0;
    uint16_t originalNSize = 0;
};

template <typename src_T = int32_t>
struct FixpipeParams {
    __aicore__ FixpipeParams()
    {
        cburstNum = DEFAULT_DATA_COPY_NBURST;
        burstLen = 1;
        srcStride = DEFAULT_DATA_COPY_STRIDE;
        dstStride = DEFAULT_DATA_COPY_STRIDE;
        reluEn = false;
        unitFlag = 0;
    }

    __aicore__ FixpipeParams(const uint16_t count, const uint16_t len, const uint16_t srcStrideIn,
        const uint32_t dstStrideIn)
    {
        cburstNum = count;
        burstLen = len;
        dstStride = dstStrideIn;
        srcStride = srcStrideIn;
    }

    uint16_t cburstNum = 0;
    uint16_t burstLen = 0;
    uint32_t dstStride = 0;
    uint16_t srcStride = 0;
    // extend param
    QuantParams quantParams;
    bool reluEn = false;
    Nz2NdParams nz2ndParams;
    uint8_t unitFlag = 0;
};
} // namespace AscendC
#endif // ASCENDC_MODULE_UTILS_STRUCT_DMA_PARAMS_H