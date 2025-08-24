/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LIB_NORMALIZATION_DEEPNORM_TILING_H
#define LIB_NORMALIZATION_DEEPNORM_TILING_H
#include "graph/tensor.h"
#include "deepnorm_tilingdata.h"
namespace AscendC {
/*!
 * \brief calculate max and min tmp buffer size for DeepNorm interface.
 * \param [in] srcShape: input shape
 * \param [in] typeSize: data type size: sizeof(TYPE)
 * \param [in] isReuseSource: indicate whether to reuse source tensor.
 *             When enable isReuseSource, src tensor will be used as tmp buffer for calculation.
 * \param [in] isBasicBlock: indicate whether enable basicBlock.
 *             When enable basicBlock, better performance will be achived and more tmp buffer will be needed
 * \param [out] maxValue: max size required for tmp buffer
 * \param [out] minValue: min size required for tmp buffer
 * \return flag for whether the tmp buffer size is calculated successfully
 *         If src shape is illegal for basic block, it will return false.
 */
bool GetDeepNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    const bool isBasicBlock, uint32_t& maxValue, uint32_t& minValue);

/*!
 * \brief calculate tiling params for DeepNorm interface
 * \note stackBufferSize should be greater than min tmpSize from GetBatchNormMaxMinTmpSize
 * \param [in] srcShape: input shape [B, S, H]
 * \param [in] originSrcShape: inputShape before 32B alignment [B, S, originH]. 0 < originH <= H
 *             When isBasicBlock = true, originH must equal H
 * \param [in] stackBufferSize: input stack buffer size in uint of Byte, used as tmp buffer size for tiling
 * \param [in] typeSize: data type size: sizeof(TYPE)
 * \param [in] isReuseSource: indicate whether intermediate variables can reuse the input memory
 * \param [in] isBasicBlock: indicate whether enable basicBlock.
 *             When enable basicBlock, better performance will be achived and more tmp buffer will be needed
 * \param [out] tiling: DeepNorm tiling
 * \return Flag for whether the tiling is calculated successfully. If src shape and origin src shape is illegal or
 *         input stackBufferSize is not big enough, it will return false.
 */
bool GetDeepNormTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, const bool isReuseSource, const bool isBasicBlock, optiling::DeepNormTiling& tiling);
}
#endif // LIB_NORMALIZATION_DEEPNORM_TILING_H
