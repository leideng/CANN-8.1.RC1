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
 * \file normalize_tiling.h
 * \brief
 */
#ifndef LIB_NORMALIZATION_NORMALIZE_TILING_H
#define LIB_NORMALIZATION_NORMALIZE_TILING_H
#include "graph/tensor.h"
namespace AscendC {
/*!
 * \brief calculate max and min tmp buffer size for Normalize interface.
 * \param [in] srcShape: input shape
 * \param [in] typeSizeU: data type size: sizeof(U)
 * \param [in] typeSizeT: data type size: sizeof(T)
 * \param [in] isReuseSource: indicate whether to reuse source tensor. Reserved paramater.
 * \param [in] isComputeRstd: indicate whether to calculate rstd. Only support true now.
 * \param [in] isOnlyOutput: indicate whether to only output normalized result y. Only support false now.
 * \param [out] maxValue: max size required for tmp buffer
 * \param [out] minValue: min size required for tmp buffer
 * \return flag for whether the tmp buffer size is calculated successfully
 */
void GetNormalizeMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSizeU, const uint32_t typeSizeT,
    const bool isReuseSource, const bool isComputeRstd, const bool isOnlyOutput, uint32_t& maxValue, uint32_t& minValue);
}
#endif // LIB_NORMALIZATION_NORMALIZE_TILING_H
