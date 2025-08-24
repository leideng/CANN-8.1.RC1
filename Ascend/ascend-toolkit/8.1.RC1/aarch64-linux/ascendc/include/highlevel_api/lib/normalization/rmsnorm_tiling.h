/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LIB_NORMALIZATION_RMSNORM_TILING_H
#define LIB_NORMALIZATION_RMSNORM_TILING_H
#include "graph/tensor.h"
#include "rmsnorm_tilingdata.h"
namespace AscendC {
/*!
 * \brief calculate max and min tmp buffer size for rmsnorm interface.
   tmp buffer size is a input for GetRmsNormTilingInfo
 *
 * \note The returned set may be smaller than set that
 *       contains all possible values of v that satisfies the bound.
 *
 * \param [in] srcShape input shape
 * \param [in] typeSize data type size: sizeof(TYPE)
 * \param [out] maxValue max size of tmp buffer
 * \param [out] minValue min size of tmp buffer
 * \param [in] isBasicBlock indicate whether enable basicBlock.
   When enable basicBlock, better performance will be achived and more tmp buffer will be needed
 * \return flag for whether the tmp buffer size is calculated successfully
           if src shape is illeagl for basic block, it will return false.
 */
bool GetRmsNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue,
    uint32_t& minValue, const bool isBasicBlock = false);

/*!
 * \brief calculate tiling params for rmsnorm interface
 *
 * \note stackBufferByteSize should be greater than min tmpSize from GetRmsNormMaxMinTmpSize
  *
 * \param [in] srcShape input shape
 * \param [in] originSrcShape data type size: sizeof(TYPE)
 * \param [in] stackBufferByteSize input stack buffer size in uint of Byte, used as tmp buffer size for tiling
 * \param [in] typeSize data type size: sizeof(TYPE)
 * \param [out] tiling RmsNorm tiling
 * \param [in] isBasicBlock indicate whether enable basicBlock.
   When enable basicBlock, better performance will be achived and more tmp buffer will be needed
 * \return flag for whether the tiling is calculated successfully
   if src shape and origin src shape is illeagl or input stackBufferByteSize is not big enough, it will return false.
 */
bool GetRmsNormTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape,
    const uint32_t stackBufferByteSize, const uint32_t typeSize, optiling::RmsNormTiling& tiling,
    const bool isBasicBlock = false);
}
#endif // LIB_NORMALIZATION_RMSNORM_TILING_H
