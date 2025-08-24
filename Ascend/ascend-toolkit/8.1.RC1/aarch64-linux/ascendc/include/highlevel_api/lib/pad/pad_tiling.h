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
 * \file pad_tiling.h
 * \brief
 */

#ifndef LIB_PAD_PAD_TILING_H
#define LIB_PAD_PAD_TILING_H

#include "graph/tensor.h"
#include "pad_tilingdata.h"
#include "tiling/platform/platform_ascendc.h"
namespace AscendC {
/*!
 * \brief calculate max and min tmp buffer size for Pad interface.
   tmp buffer size is a input for PadTilingFunc
 *
 * \param [in] srcShape input shape
 * \param [in] typeSize data type size: sizeof(TYPE)
 * \param [out] maxValue max size of tmp buffer
 * \param [out] minValue min size of tmp buffer
 */
void GetPadMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue,
    uint32_t& minValue);

/*!
 * \brief calculate tiling params for Pad interface
 *
 * \note stackBufferSize should be greater than min tmpSize from GetPadMaxMinTmpSize
 *
 * \param [in] srcShape input shape
 * \param [in] originSrcShape data type size: sizeof(TYPE)
 * \param [in] stackBufferSize input stack buffer size in uint of Byte, used as tmp buffer size for tiling
 * \param [in] typeSize data type size: sizeof(TYPE)
 * \param [out] tiling Pad tiling
 */
void PadTilingFunc(const ge::Shape srcShape, const ge::Shape oriSrcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, optiling::PadTiling& tiling);

/*!
 * \brief calculate max and min tmp buffer size for UnPad interface.
   tmp buffer size is a input for UnPadTilingFunc
 *
 * \param [in] ascendcPlatform ascendc platform
 * \param [in] srcShape input shape
 * \param [in] typeSize data type size: sizeof(TYPE)
 * \param [out] maxValue max size of tmp buffer
 * \param [out] minValue min size of tmp buffer
 */
void GetUnPadMaxMinTmpSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const ge::Shape& srcShape,
    const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue);

/*!
 * \brief calculate tiling params for UnPad interface
 *
 * \note stackBufferSize should be greater than min tmpSize from GetUnPadMaxMinTmpSize
 *
 * \param [in] srcShape input shape
 * \param [in] originSrcShape data type size: sizeof(TYPE)
 * \param [in] stackBufferSize input stack buffer size in uint of Byte, used as tmp buffer size for tiling
 * \param [in] typeSize data type size: sizeof(TYPE)
 * \param [out] tiling UnPad tiling
 */
void UnPadTilingFunc(const ge::Shape srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
    optiling::UnPadTiling& tiling);
}
#endif // LIB_PAD_PAD_TILING_H
