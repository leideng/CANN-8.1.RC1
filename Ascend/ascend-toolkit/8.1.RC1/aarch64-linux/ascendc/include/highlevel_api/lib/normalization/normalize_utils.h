/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file normalize_utils.h
 * \brief
 */
#ifndef LIB_NORMALIZATION_NORMALIZE_UTILS_H
#define LIB_NORMALIZATION_NORMALIZE_UTILS_H

namespace AscendC {

enum class ReducePattern : uint32_t {
    AR = 0,
    RA = 1,
};

struct NormalizeConfig {
    ReducePattern reducePattern = ReducePattern::AR;
    int32_t aLength = -1;
    bool isNoBeta = false;
    bool isNoGamma = false;
    bool isOnlyOutput = false;
};

__aicore__ constexpr NormalizeConfig GetNormalizeConfig(bool isNoBeta, bool isNoGamma)
{
    return {.reducePattern = ReducePattern::AR,
        .aLength = -1,
        .isNoBeta = isNoBeta,
        .isNoGamma = isNoGamma,
        .isOnlyOutput = false};
}

constexpr NormalizeConfig NLCFG_NORM = GetNormalizeConfig(false, false);

constexpr NormalizeConfig NLCFG_NOBETA = GetNormalizeConfig(true, false);

constexpr NormalizeConfig NLCFG_NOGAMMA = GetNormalizeConfig(false, true);

constexpr NormalizeConfig NLCFG_NOOPT = GetNormalizeConfig(true, true);

}; // namespace AscendC
#endif // LIB_NORMALIZATION_NORMALIZE_UTILS_H