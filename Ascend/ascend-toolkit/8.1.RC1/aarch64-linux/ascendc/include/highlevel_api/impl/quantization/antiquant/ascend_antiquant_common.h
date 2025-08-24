/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ascend_antiquant_common.h
 * \brief
 */
#ifndef IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_COMMON_H
#define IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_COMMON_H

namespace AscendC {
constexpr uint32_t ANTIQUANT_TWO = 2;
constexpr uint32_t ANTIQUANT_FOUR = 4;
constexpr uint32_t ANTIQUANT_BRCB_BASE = 8;    // BRCB need at least 8 number
constexpr uint32_t ANTIQUANT_MIN_METHOD2 = 80; // min tmpBuffersize must be 80 * N
constexpr uint32_t ANTIQUANT_SINGLE_N_SIZE = 64;
constexpr uint32_t ANTIQUANT_SINGLE_N_SIZE_BF16 = 64;
constexpr uint32_t ANTIQUANT_SINGLE_N_SIZE_FP16 = 128;
constexpr uint32_t ANTIQUANT_MAX_K = 255;
constexpr uint32_t MAX_K_FOR_FP16_BRCB = 4096;

struct AntiQuantShapeInfo {
    uint32_t offsetHeight{0};
    uint32_t offsetWidth{0};
    uint32_t scaleHeight{0};
    uint32_t scaleWidth{0};
};

} // namespace AscendC
#endif // IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_COMMON_H
