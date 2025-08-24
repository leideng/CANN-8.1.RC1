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
 * \file digamma_tiling.h
 * \brief
 */
#ifndef LIB_MATH_DIGAMMA_TILING_H
#define LIB_MATH_DIGAMMA_TILING_H

#include <cstdint>
#include "graph/tensor.h"

namespace AscendC {
/*
 * @ingroup GetDigammaMaxMinTmpSize
 * @brief Get Digamma api calculate need max and min temporary local space size.
 * @param [in] srcShape: src tensor shape.
 * @param [in] typeSize: src and dst tensor dtype size.
 * @param [in] isReuseSource: whether temporary variables can reuse the input memory.
 * @param [out] maxValue: Digamma api calculate need max temporary local space size.
 * @param [out] minValue: Digamma api calculate need min temporary local space size.
 */
void GetDigammaMaxMinTmpSize(const ge::Shape &srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t &maxValue, uint32_t &minValue);

/*
 * @ingroup GetDigammaTmpBufferFactorSize
 * @brief Get the size relationship between the temporary space used by the digamma API calculation and
          the input tensor space, and the extra space used.
 * @param [in] typeSize: src and dst tensor dtype size.
 * @param [out] maxLiveNodeCount: The maximum temporary memory used by the Digamma API for
                                  calculation is several times the input size.
 * @param [out] extraBuffer: The extra memory space used by the Digamma API for calculation.
 */
void GetDigammaTmpBufferFactorSize(const uint32_t typeSize, uint32_t &maxLiveNodeCount, uint32_t &extraBuffer);
} // namespace AscendC

#endif // LIB_MATH_DIGAMMA_TILING_H