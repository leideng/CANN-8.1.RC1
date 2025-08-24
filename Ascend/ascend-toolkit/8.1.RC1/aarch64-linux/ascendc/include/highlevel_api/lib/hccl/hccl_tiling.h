/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file hccl_tiling.h
* \brief
*/
#ifndef LIB_HCCL_HCCL_TILING_H
#define LIB_HCCL_HCCL_TILING_H

#include "hccl_tilingdata.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {

class Mc2CcTilingConfig {
public:
    explicit Mc2CcTilingConfig(const std::string &groupName, uint32_t opType, const std::string &algConfig,
                               uint32_t reduceType = 0);
    virtual ~Mc2CcTilingConfig();

public:
    uint32_t GetTiling(::Mc2InitTiling &tiling);
    uint32_t GetTiling(::Mc2CcTiling &tiling);

public:
    uint32_t SetOpType(uint32_t opType);
    uint32_t SetGroupName(const std::string &groupName);
    uint32_t SetAlgConfig(const std::string &algConfig);
    uint32_t SetReduceType(uint32_t reduceType);
    uint32_t SetStepSize(uint8_t stepSize);
    uint32_t SetSkipLocalRankCopy(uint8_t skipLocalRankCopy);
    uint32_t SetSkipBufferWindowCopy(uint8_t skipBufferWindowCopy);
    uint32_t SetDebugMode(uint8_t debugMode);

private:
    uint32_t opType_;
    std::string groupName_;
    std::string algConfig_;
    uint32_t reduceType_ = 0;
    uint8_t stepSize_ = 0;
    uint8_t skipLocalRankCopy_ = 0;
    uint8_t skipBufferWindowCopy_ = 0;
    uint8_t debugMode_ = 0;

    uint64_t initTilingAddr_ = 0;
};
} // namespace AscendC
#endif