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
 * \file frac_tiling.h
 * \brief
 */
#ifndef LIB_MATH_FRAC_TILING_H
#define LIB_MATH_FRAC_TILING_H
#include <cstdint>

#include "graph/tensor.h"
namespace AscendC {
/*!
 * \brief This interface is used to obtain the maximum and minimum temporary space reserved or applied.
 * The developer selects a proper space size based on this range as the tiling parameter.
 *
 * \param [in] srcShape, input shape information
 * \param [in] typeSize, size of the input data type, in bytes
 * \param [in] isReuseSource, whether to reuse the input space of the source operand
 * \param [out] maxValue, maximum temporary space required
 * \param [out] minValue, minimum temporary space required
 */
void GetFracMaxMinTmpSize(const ge::Shape &srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t &maxValue, uint32_t &minValue);

/*!
 * \brief The calculation of the Frac interface requires the developer to reserve or apply for temporary space.
 * It's used to derive the size of elements per calculation iteration based on remain free space size.
 * iterationSize = (remainFreeSapce - extraBuf) / maxLivedNodeCnt / typeSize
 *
 * \param [in] typeSize, size of the input data type, in bytes
 * \param [out] maxLiveNodeCnt, the multiple of the maximum temporary space to the input occupied space
 * \param [out] extraBuf, the size of the extra temporary space
 */
void GetFracTmpBufferFactorSize(const uint32_t typeSize, uint32_t &maxLiveNodeCnt, uint32_t &extraBuf);
} // namespace AscendC
#endif // LIB_MATH_FRAC_TILING_H