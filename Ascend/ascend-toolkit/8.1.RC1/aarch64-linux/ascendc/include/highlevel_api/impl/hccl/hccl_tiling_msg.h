/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_tiling_msg.h
 * \brief
 */
#ifndef IMPL_HCCL_HCCL_TILING_MSG_H
#define IMPL_HCCL_HCCL_TILING_MSG_H

namespace AscendC {

constexpr uint32_t INIT_TILING_SIZE = 48U;
constexpr uint32_t CC_TILING_SIZE = 280U;
constexpr uint32_t INIT_TILING_OFFSET = 8U;
constexpr uint32_t INIT_TILING_VERSION = 100U;
constexpr uint32_t GROUP_NAME_SIZE = 128U;
constexpr uint32_t ALG_CONFIG_SIZE = 128U;
constexpr uint8_t MC2_INIT_TILING_VERSION = 0U;
constexpr uint8_t MC2_CC_TILING_VERSION = 0U;

struct Mc2InitTilingInner {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[INIT_TILING_OFFSET];
    uint8_t debugMode;
    uint8_t preparePosition;
    char reserved[22];
};

struct Mc2CcTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[12];
    char groupName[GROUP_NAME_SIZE];
    char algConfig[ALG_CONFIG_SIZE];
    uint32_t opType;
    uint32_t reduceType;
};

enum HcclTilingStatus {
    TILING_SUCCESS = 0,
    TILING_FAILED = 1,
    HCCL_TILING_STATUS_RESERVED
};
} // namespace AscendC

#endif // IMPL_HCCL_HCCL_TILING_MSG_H