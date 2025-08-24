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
 * \file selectwithbytesmask_tiling.h
 * \brief
 */
#ifndef LIB_SELECT_SELECT_WITH_BYTES_MASK_TILING_H
#define LIB_SELECT_SELECT_WITH_BYTES_MASK_TILING_H
#include <cstdint>
#include <algorithm>

#include "graph/tensor.h"
namespace AscendC {
/* !
 * \brief This interface is used to obtain the minimum temporary space reserved or applied.
 * The developer selects a proper space size based on this range as the tiling parameter.
 *
 * \note one of the sources must be scalar type, whose shape size is either 0 or 1, mask tensor can't be scalar.
 * \param [in] src0Shape, input source0 shape information
 * \param [in] src1Shape, input source1 shape information
 * \param [in] srcTypeSize, size of the input source data type in bytes
 * \param [in] maskShape, input mask shape information
 * \param [in] maskTypeSize, size of the input mask data type in bytes
 * \param [in] isReuseMask, whether to reuse the input space of the mask operand
 * \return minValue, minimum temporary space required
 */
uint32_t GetSelectWithBytesMaskMinTmpSize(
    const ge::Shape &src0Shape, const ge::Shape &src1Shape, const uint32_t srcTypeSize, const ge::Shape &maskShape,
    const uint32_t maskTypeSize, const bool isReuseMask);

/* !
 * \brief This interface is used to obtain the maximum temporary space reserved or applied.
 * The developer selects a proper space size based on this range as the tiling parameter.
 *
 * \note one of the sources must be scalar type, whose shape size is either 0 or 1, mask tensor can't be scalar.
 * \param [in] src0Shape, input source0 shape information
 * \param [in] src1Shape, input source1 shape information
 * \param [in] srcTypeSize, size of the input source data type in bytes
 * \param [in] maskShape, input mask shape information
 * \param [in] maskTypeSize, size of the input mask data type in bytes
 * \param [in] isReuseMask, whether to reuse the input space of the mask operand
 * \return maxValue, maximum temporary space required
 */
uint32_t GetSelectWithBytesMaskMaxTmpSize(
    const ge::Shape &src0Shape, const ge::Shape &src1Shape, const uint32_t srcTypeSize, const ge::Shape &maskShape,
    const uint32_t maskTypeSize, const bool isReuseMask);

/* !
 * \brief This interface is used to obtain the maximum and minimum temporary space reserved or applied.
 * The developer selects a proper space size based on this range as the tiling parameter.
 *
 * \note one of the sources must be scalar type, whose shape size is either 0 or 1, mask tensor can't be scalar.
 * \param [in] src0Shape, input source0 shape information
 * \param [in] src1Shape, input source1 shape information
 * \param [in] srcTypeSize, size of the input source data type in bytes
 * \param [in] maskShape, input mask shape information
 * \param [in] maskTypeSize, size of the input mask data type in bytes
 * \param [in] isReuseMask, whether to reuse the input space of the mask operand
 * \param [out] maxValue, maximum temporary space required
 * \param [out] minValue, minimum temporary space required
 */
void GetSelectWithBytesMaskMaxMinTmpSize(const ge::Shape &src0Shape, const ge::Shape &src1Shape,
    const uint32_t srcTypeSize, const ge::Shape &maskShape, const uint32_t maskTypeSize, const bool isReuseMask,
    uint32_t &maxValue, uint32_t &minValue);
} // namespace AscendC
#endif // LIB_SELECT_SELECT_WITH_BYTES_MASK_TILING_H