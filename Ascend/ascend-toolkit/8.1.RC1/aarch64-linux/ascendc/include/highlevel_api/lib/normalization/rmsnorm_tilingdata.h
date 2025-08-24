/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIB_NORMALIZATION_RMSNORM_TILINGDATA_H
#define LIB_NORMALIZATION_RMSNORM_TILINGDATA_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RmsNormTiling)
    TILING_DATA_FIELD_DEF(uint32_t, bLength);
    TILING_DATA_FIELD_DEF(uint32_t, sLength);
    TILING_DATA_FIELD_DEF(uint32_t, hLength);
    TILING_DATA_FIELD_DEF(uint32_t, originalHLength);
    TILING_DATA_FIELD_DEF(float, reciprocalOfHLength);
    TILING_DATA_FIELD_DEF(uint32_t, mainBshLength);
    TILING_DATA_FIELD_DEF(uint32_t, mainBsLength);
    TILING_DATA_FIELD_DEF(uint32_t, mainBsLengthAlign);
    TILING_DATA_FIELD_DEF(uint32_t, loopRound);
    TILING_DATA_FIELD_DEF(uint32_t, inputTailPos);
    TILING_DATA_FIELD_DEF(uint32_t, tailBshLength);
    TILING_DATA_FIELD_DEF(uint32_t, tailBsLength);
END_TILING_DATA_DEF;
}
#endif // LIB_NORMALIZATION_RMSNORM_TILINGDATA_H
