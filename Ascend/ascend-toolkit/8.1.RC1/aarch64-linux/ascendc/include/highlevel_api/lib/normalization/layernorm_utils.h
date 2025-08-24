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
 * \file layernorm_utils.h
 * \brief
 */
#ifndef LIB_NORMALIZATION_LAYERNORM_UTILS_H
#define LIB_NORMALIZATION_LAYERNORM_UTILS_H

namespace AscendC {

struct LayerNormConfig {
    bool isNoBeta = false;
    bool isNoGamma = false;
    bool isOnlyOutput = false;
};

__aicore__ constexpr LayerNormConfig GetLayerNormNormalConfig()
{
    return {.isNoBeta = false, .isNoGamma = false, .isOnlyOutput = false};
}

constexpr LayerNormConfig LNCFG_NORM = GetLayerNormNormalConfig();

struct WelfordUpdateConfig {
    __aicore__ constexpr WelfordUpdateConfig(const bool isInplaceIn): isInplace(isInplaceIn) {}
    bool isInplace = false;
};

constexpr WelfordUpdateConfig WFUPDATE_DEFAULT_CFG = {false};

}; // namespace AscendC
#endif // LIB_NORMALIZATION_LAYERNORM_UTILS_H