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
 * \file swish_tiling.h
 * \brief
 */
#ifndef LIB_ACTIVATION_SWISH_TILING_H
#define LIB_ACTIVATION_SWISH_TILING_H
#include "graph/tensor.h"
#include "register/tilingdata_base.h"

namespace AscendC {
inline void GetSwishTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& max,
    uint32_t& min)
{
    (void)srcShape;
    (void)typeSize;
    (void)isReuseSource;
    max = 0;
    min = 0;
}
} // namespace AscendC
#endif