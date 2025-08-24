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
 * \file cumsum_tiling.h
 * \brief
 */
#ifndef LIB_MATH_CUMSUM_TILING_H
#define LIB_MATH_CUMSUM_TILING_H
#include <cstdint>

#include "graph/tensor.h"
namespace AscendC {
/*
 * @ingroup GetCumSumMaxMinTmpSize
 * @brief get cumsum api calculate need max and min temporary local space size
 * @param [in] srcShape : src tensor shape
 * @param [in] typeSize : src tensor dtype size
 * @param [in] isLastAxis : whether to operate along the last axis
 * @param [in] isReuseSource : whether to reuse the src Tensor
 * @return max temporary local space size
 * @return min temporary local space size
 */
void GetCumSumMaxMinTmpSize(const ge::Shape &srcShape, const uint32_t typeSize, const bool isLastAxis,
    const bool isReuseSource, uint32_t &maxValue, uint32_t &minValue);
}  // namespace AscendC
#endif