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
 * \file confusion_transpose_tiling.h
 * \brief
 */
#ifndef LIB_TRANSPOSE_CONFUSION_TRANSPOSE_TILING_H
#define LIB_TRANSPOSE_CONFUSION_TRANSPOSE_TILING_H
#include "graph/tensor.h"
#include "confusion_transpose_tilingdata.h"
namespace AscendC {
constexpr uint32_t TWO_TIMES = 2;
constexpr uint32_t BLOCK_CUBE = 16;
constexpr uint32_t ONE_BLK_SIZE = 32;
constexpr int32_t CUBE_MAX_SIZE = 256;
/*!
 * \brief calculate max and min tmp buffer size for ConfusionTranspose interface.
   tmp buffer size is a input for GetConfusionTransposeTilingInfo
 *
 * \param [in] srcShape input shape
 * \param [in] typeSize data type size: sizeof(TYPE)
 * \param [in] transposeTypeIn transpose type.
 * \param [out] maxValue max size of tmp buffer
 * \param [out] minValue min size of tmp buffer
 */
void GetConfusionTransposeMaxMinTmpSize(const ge::Shape &srcShape, const uint32_t typeSize,
    const uint32_t transposeTypeIn, uint32_t &maxValue, uint32_t &minValue);

/*!
 * \brief calculate tiling params for ConfusionTranspose interface
 *
 * \note stackBufferSize should be greater than min tmpSize from GetConfusionTransposeMaxMinTmpSize
 *
 * \param [in] srcShape input shape
 * \param [in] stackBufferSize input stack buffer size in uint of Byte, used as tmp buffer size for tiling
 * \param [in] typeSize data type size: sizeof(TYPE)
 * \param [in] transposeTypeIn transpose type.
 * \param [out] tiling ConfusionTranspose tiling
 */
void GetConfusionTransposeTilingInfo(const ge::Shape &srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
    const uint32_t transposeTypeIn, optiling::ConfusionTransposeTiling &tiling);

void GetConfusionTransposeOnlyTilingInfo(const ge::Shape &srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, optiling::ConfusionTransposeTiling &tiling);
}
#endif // LIB_TRANSPOSE_CONFUSION_TRANSPOSE_TILING_H
