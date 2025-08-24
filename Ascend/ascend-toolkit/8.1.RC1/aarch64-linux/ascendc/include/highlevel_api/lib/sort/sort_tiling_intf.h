/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LIB_SORT_SORT_TILING_INTF_H
#define LIB_SORT_SORT_TILING_INTF_H

#include <cstdint>

#include "tiling/platform/platform_ascendc.h"

namespace AscendC {
constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_V200 = 8;
constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_HALF_V220 = 4;
constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_FLOAT_V220 = 2;

/* **************************************** GetConcatTmpSize ****************************************** */
/*
 * @ingroup GetConcatTmpSize
 * @brief get concat tmp buffer size
 * @param [in] ascendcPlatform ascendc platform infomation
 * @param [in] elemCount element count number
 * @param [in] dataTypeSize data size number
 */
inline uint32_t GetConcatTmpSize(const platform_ascendc::PlatformAscendC &ascendcPlatform, const uint32_t elemCount,
    const uint32_t dataTypeSize)
{
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion == platform_ascendc::SocVersion::ASCEND910B) {
        return 0;
    } else {
        return elemCount * REGION_PROPOSAL_DATA_SIZE_V200 * dataTypeSize;
    }
}

/* **************************************** GetSortTmpSize ****************************************** */
/*
 * @ingroup GetSortTmpSize
 * @brief get sort tmp buffer size
 * @param [in] ascendcPlatform ascendc platform infomation
 * @param [in] elemCount element count number
 * @param [in] dataTypeSize data size number
 */
inline uint32_t GetSortTmpSize(const platform_ascendc::PlatformAscendC &ascendcPlatform, const uint32_t elemCount,
    const uint32_t dataTypeSize)
{
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion == platform_ascendc::SocVersion::ASCEND910B) {
        if (dataTypeSize == sizeof(float)) {
            return elemCount * REGION_PROPOSAL_DATA_SIZE_FLOAT_V220 * dataTypeSize;
        } else {
            return elemCount * REGION_PROPOSAL_DATA_SIZE_HALF_V220 * dataTypeSize;
        }
    } else {
        return elemCount * REGION_PROPOSAL_DATA_SIZE_V200 * dataTypeSize;
    }
}
} // namespace AscendC
#endif // LIB_SORT_SORT_TILING_INTF_H