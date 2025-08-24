/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file groupnorm_tiling.h
 * \brief
 */

#ifndef LIB_NORMALIZATION_GROUPNORM_TILING_H
#define LIB_NORMALIZATION_GROUPNORM_TILING_H
#include "graph/tensor.h"
#include "groupnorm_tilingdata.h"
namespace AscendC {
/*!
 * \brief calculate max and min tmp buffer size for GroupNorm interface.
 * \param [in] srcShape: input shape
 * \param [in] typeSize: data type size: sizeof(TYPE)
 * \param [in] isReuseSource: indicate whether to reuse source tensor.
 *             When enable isReuseSource, src tensor will be used as tmp buffer for calculation.
 * \param [in] groupNum: number of groups to separate the channels into
 * \param [out] maxValue: max size required for tmp buffer
 * \param [out] minValue: min size required for tmp buffer
 * \return flag for whether the tmp buffer size is calculated successfully
 *         If src shape is illegal for basic block, it will return false.
 */
void GetGroupNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    const uint32_t groupNum, uint32_t& maxValue, uint32_t& minValue);

/*!
 * \brief calculate tiling params for GroupNorm interface
 *
 * \note stackBufferSize should be greater than min tmpSize from GetGroupNormMaxMinTmpSize
  *
 * \param [in] srcShape input shape
 * \param [in] stackBufferSize input stack buffer size in uint of Byte, used as tmp buffer size for tiling
 * \param [in] typeSize data type size: sizeof(TYPE)
 * \param [in] isReuseSource indicate whether intermediate variables can reuse the input memory
 * \param [in] groupNum: number of groups to separate the channels into
 * \param [out] tiling GroupNorm tiling
 * \return flag for whether the tiling is calculated successfully
   if src shape and origin src shape is illeagl or input stackBufferSize is not big enough, it will return false.
 */
void GetGroupNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
    const bool isReuseSource, const uint32_t groupNum, optiling::GroupNormTiling& tiling);
}
#endif // LIB_NORMALIZATION_GROUPNORM_TILING_H
