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
 * \file softmax_utils.h
 * \brief
 */
#ifndef LIB_ACTIVATION_SOFTMAX_UTILS_H
#define LIB_ACTIVATION_SOFTMAX_UTILS_H

namespace AscendC {

enum class SoftmaxMode {
    SOFTMAX_NORMAL = 0,
    SOFTMAX_OUTPUT_WITHOUT_BRC = 1,
};

struct SoftmaxConfig {
    __aicore__ constexpr SoftmaxConfig(const bool isCheckTilingIn)
    {
        isCheckTiling = isCheckTilingIn;
    }
    __aicore__ constexpr SoftmaxConfig(const bool isCheckTilingIn, const uint32_t oriSrcMIn, const uint32_t oriSrcKIn)
    {
        isCheckTiling = isCheckTilingIn;
        oriSrcM = oriSrcMIn;
        oriSrcK = oriSrcKIn;
    }
    __aicore__ constexpr SoftmaxConfig(const bool isCheckTilingIn, const uint32_t oriSrcMIn, const uint32_t oriSrcKIn, const enum SoftmaxMode modeIn)
    {
        isCheckTiling = isCheckTilingIn;
        oriSrcM = oriSrcMIn;
        oriSrcK = oriSrcKIn;
        mode = modeIn;
    }
    // to judge if match or not of input shape and tiling, if not match, api will recompute tiling, default to judge
    bool isCheckTiling = true;
    uint32_t oriSrcM = 0;
    uint32_t oriSrcK = 0;
    SoftmaxMode mode = SoftmaxMode::SOFTMAX_NORMAL;
};

constexpr SoftmaxConfig SOFTMAX_DEFAULT_CFG = { true, 0, 0 , SoftmaxMode::SOFTMAX_NORMAL};

}; // namespace AscendC
#endif // LIB_ACTIVATION_SOFTMAX_UTILS_H