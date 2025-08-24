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
 * \file ascend_quant_utils.h
 * \brief
 */
#ifndef LIB_QUANTIZATION_ASCEND_QUANT_UTILS_H
#define LIB_QUANTIZATION_ASCEND_QUANT_UTILS_H

namespace AscendC {

struct AscendQuantConfig {
    __aicore__ constexpr AscendQuantConfig(const uint32_t calcCount, const uint32_t offsetCount,
        const uint32_t scaleCount, const uint32_t workLocalSize): calcCount(calcCount), offsetCount(offsetCount),
        scaleCount(scaleCount), workLocalSize(workLocalSize) {}
    uint32_t calcCount = 0;
    uint32_t offsetCount = 0;
    uint32_t scaleCount = 0;
    uint32_t workLocalSize = 0;
};

constexpr AscendQuantConfig ASCEND_QUANT_DEFAULT_CFG = {0, 0, 0, 0};

}; // namespace AscendC
#endif // LIB_QUANTIZATION_ASCEND_QUANT_UTILS_H