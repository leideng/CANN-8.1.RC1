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
 * \file reglu_tiling.h
 * \brief
 */
#ifndef LIB_REGLU_REGLU_TILING_H
#define LIB_REGLU_REGLU_TILING_H
#include <cstdint>

#include "graph/tensor.h"

namespace AscendC {
 /*
 * This interface is used to obtain the maximum and minimum temporary space reserved or applied.
 * The developer selects a proper space size based on this range as the tiling parameter.
 * @ingroup ReGlu
 * @param [in] srcShape, input shape information
 * @param [in] typeSize, input LocalTensor typeSize, half is 2 and float is 4
 * @param [in] isReuseSource, whether to reuse the input space of the source operand
 * @param [out] maxValue, maximum temporary space required
 * @param [out] minValue, minimum temporary space required
 */
void GetReGluMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue);
} // namespace AscendC
#endif // LIB_REGLU_REGLU_TILING_H