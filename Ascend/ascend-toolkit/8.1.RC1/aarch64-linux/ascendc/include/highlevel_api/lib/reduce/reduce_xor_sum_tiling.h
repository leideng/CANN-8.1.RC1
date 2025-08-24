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
 * \file reduce_xor_sum_tiling.h
 * \brief
 */
#ifndef  LIB_REDUCE_REDUCE_XOR_SUM_TILING_H
#define  LIB_REDUCE_REDUCE_XOR_SUM_TILING_H

#include <cstdint>
#include "graph/tensor.h"

namespace AscendC {
/*
 * @ingroup GetReduceXorSumMaxMinTmpSize
 * @brief Get ReduceXorSum api calculate need max and min temporary local space size.
 * @param [in] srcShape: src tensor shape.
 * @param [in] typeSize: src and dst tensor dtype size.
 * @param [in] isReuseSource: whether temporary variables can reuse the input memory.
 * @param [out] maxValue: ReduceXorSum api calculate need max temporary local space size.
 * @param [out] minValue: ReduceXorSum api calculate need min temporary local space size.
 */
void GetReduceXorSumMaxMinTmpSize(const ge::Shape &srcShape, const uint32_t typeSize, const bool isReuseSource,
                                  uint32_t &maxValue, uint32_t &minValue);
}  // namespace AscendC

#endif  // LIB_REDUCE_REDUCE_XOR_SUM_TILING_H