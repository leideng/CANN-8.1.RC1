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
 * \file broadcast_tiling.h
 * \brief
 */
#ifndef LIB_PAD_BROADCAST_TILING_H
#define LIB_PAD_BROADCAST_TILING_H

#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
namespace AscendC {
/*
 * @ingroup GetBroadCastMaxMinTmpSize
 * @brief get broadcast api calculate need max and min temporary local space size
 * @param [in] ascendcPlatform : ascendc platform infomation
 * @param [in] srcShape : src tensor shape
 * @param [in] dstShape : dst tensor shape
 * @param [in] typeSize : src and dst tensor dtype size
 * @param [in] isReuseSource : whether to reuse the src Tensor
 * @return max temporary local space size
 * @return min temporary local space size
 */
void GetBroadCastMaxMinTmpSize(const platform_ascendc::PlatformAscendC &ascendcPlatform, const ge::Shape &srcShape,
    const ge::Shape &dstShape, uint32_t typeSize, const bool isReuseSource, uint32_t &maxValue,
    uint32_t &minValue);
}  // namespace AscendC
#endif // LIB_PAD_BROADCAST_TILING_H