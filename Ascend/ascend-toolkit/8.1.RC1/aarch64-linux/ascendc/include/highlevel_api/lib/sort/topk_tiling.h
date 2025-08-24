/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LIB_SORT_TOPK_TILING_H
#define LIB_SORT_TOPK_TILING_H
#include "topk_tilingdata.h"
#include "tiling/platform/platform_ascendc.h"

namespace AscendC {

enum class TopKMode {
    TOPK_NORMAL,
    TOPK_NSMALL,
};

/*
 * @ingroup GetTopKMaxMinTmpSize
 * @brief Get TopK api calculate need max and min temporary local space size.
 * @param [in] ascendcPlatform: Information about the hardware platform.
 * @param [in] inner: Inner axis length of input data.
 * @param [in] outter: Outer axis length of input data.
 * @param [in] isReuseSource: Whether temporary variables can reuse the input memory.
 * @param [in] isInitIndex: Whether to transfer the index of the input data.
 * @param [in] mode: Normal mode or small mode.
 * @param [in] isLargest: Descending or ascending order.
 * @param [in] dataTypeSize: Input data dtype size.
 * @param [out] maxValue: TopK api calculate need max temporary local space size.
 * @param [out] minValue: TopK api calculate need min temporary local space size.
 * @return true: Succeeded in obtaining the maximum and minimum temporary space sizes.
 * @return false: Failed to obtain maximum and minimum temporary space sizes.
 */
bool GetTopKMaxMinTmpSize(const platform_ascendc::PlatformAscendC &ascendcPlatform, const int32_t inner,
    const int32_t outter, const bool isReuseSource, const bool isInitIndex, enum TopKMode mode,
    const bool isLargest, const uint32_t dataTypeSize, uint32_t &maxValue, uint32_t &minValue);

/*
 * @ingroup TopKTilingFunc
 * @brief Get the tiling information required by the Topk interface.
 * @param [in] ascendcPlatform: Information about the hardware platform.
 * @param [in] inner: Inner axis length of input data.
 * @param [in] outter: Outer axis length of input data.
 * @param [in] k: Obtain the first k maximum or minimum values and their corresponding indexes.
 * @param [in] dataTypeSize: Input data dtype size.
 * @param [in] isInitIndex: Whether to transfer the index of the input data.
 * @param [in] mode: Normal mode or small mode.
 * @param [in] isLargest: Descending or ascending order.
 * @param [out] topKTiling: Output the tiling information required by the Topk interface.
 * @return true: The values of the TopK tiling parameters are obtained successfully.
 * @return false: Failed to obtain tiling data.
 */
bool TopKTilingFunc(const platform_ascendc::PlatformAscendC &ascendcPlatform, const int32_t inner, const int32_t outter,
    const int32_t k, const uint32_t dataTypeSize, const bool isInitIndex, enum TopKMode mode, const bool isLargest,
    optiling::TopkTiling &topKTiling);

}  // namespace AscendC
#endif  // LIB_SORT_TOPK_TILING_H