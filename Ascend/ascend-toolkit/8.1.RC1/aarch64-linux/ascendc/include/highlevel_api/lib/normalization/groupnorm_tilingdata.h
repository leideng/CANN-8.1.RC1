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
 * \file groupnorm_tilingdata.h
 * \brief
 */

#ifndef LIB_NORMALIZATION_GROUPNORM_TILINGDATA_H
#define LIB_NORMALIZATION_GROUPNORM_TILINGDATA_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GroupNormTiling)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, c);
    TILING_DATA_FIELD_DEF(uint32_t, hw);
    TILING_DATA_FIELD_DEF(uint32_t, g);
    TILING_DATA_FIELD_DEF(uint32_t, d);
    TILING_DATA_FIELD_DEF(uint32_t, hwAlignSize);
    TILING_DATA_FIELD_DEF(uint32_t, dhwAlignSize);
    TILING_DATA_FIELD_DEF(uint32_t, inputXSize);
    TILING_DATA_FIELD_DEF(uint32_t, meanVarSize);
    TILING_DATA_FIELD_DEF(uint32_t, numberOfTmpBuf);
    TILING_DATA_FIELD_DEF(uint32_t, meanTmpTensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, meanTmpTensorSize);
    TILING_DATA_FIELD_DEF(uint32_t, varianceTmpTensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, varianceTmpTensorSize);
    TILING_DATA_FIELD_DEF(uint32_t, tmpBufSize);
    TILING_DATA_FIELD_DEF(uint32_t, oneTmpSize);
    TILING_DATA_FIELD_DEF(uint32_t, firstTmpStartPos);
    TILING_DATA_FIELD_DEF(uint32_t, secondTmpStartPos);
    TILING_DATA_FIELD_DEF(uint32_t, thirdTmpStartPos);
    TILING_DATA_FIELD_DEF(uint32_t, loopRound);
    TILING_DATA_FIELD_DEF(uint32_t, inputRoundSize);
    TILING_DATA_FIELD_DEF(uint32_t, inputTailSize);
    TILING_DATA_FIELD_DEF(uint32_t, inputTailPos);
    TILING_DATA_FIELD_DEF(uint32_t, meanVarRoundSize);
    TILING_DATA_FIELD_DEF(uint32_t, meanVarTailSize);
    TILING_DATA_FIELD_DEF(uint32_t, meanVarTailPos);
    TILING_DATA_FIELD_DEF(uint32_t, bshCurLength);
    TILING_DATA_FIELD_DEF(uint32_t, bsCurLength);
    TILING_DATA_FIELD_DEF(float, factor);
    TILING_DATA_FIELD_DEF(bool, smallShape);
END_TILING_DATA_DEF;
}
#endif // LIB_NORMALIZATION_GROUPNORM_TILINGDATA_H