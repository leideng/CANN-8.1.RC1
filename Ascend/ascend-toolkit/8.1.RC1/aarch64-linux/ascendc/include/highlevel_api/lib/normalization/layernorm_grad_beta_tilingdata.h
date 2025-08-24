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
 * \file layernorm_grad_beta_tilingdata.h
 * \brief
 */

#ifndef LIB_NORMALIZATION_LAYERNORM_GRAD_BETA_TILINGDATA_H
#define LIB_NORMALIZATION_LAYERNORM_GRAD_BETA_TILINGDATA_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LayerNormGradBetaTiling)
    TILING_DATA_FIELD_DEF(uint32_t, stackBufferSize);
    TILING_DATA_FIELD_DEF(uint32_t, bLength);
    TILING_DATA_FIELD_DEF(uint32_t, sLength);
    TILING_DATA_FIELD_DEF(uint32_t, hLength);
    TILING_DATA_FIELD_DEF(uint32_t, originalHLength);
    TILING_DATA_FIELD_DEF(uint32_t, bshLength);
    TILING_DATA_FIELD_DEF(uint32_t, bsLength);
    TILING_DATA_FIELD_DEF(uint32_t, oneCalSize);
    TILING_DATA_FIELD_DEF(uint32_t, numberOfTmpBuf);
    TILING_DATA_FIELD_DEF(uint32_t, loopRound);
    TILING_DATA_FIELD_DEF(uint32_t, inputTailSize);
    TILING_DATA_FIELD_DEF(uint32_t, inputTailPos);
    TILING_DATA_FIELD_DEF(uint32_t, bsTailSize);
    TILING_DATA_FIELD_DEF(uint32_t, bshCurLength);
    TILING_DATA_FIELD_DEF(uint32_t, bsCurLength);
    TILING_DATA_FIELD_DEF(uint32_t, gammaTempTensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, betaTempTensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, inputDyTmpTensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, resForGammaTmpTensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;
}
#endif // LIB_NORMALIZATION_LAYERNORM_GRAD_BETA_TILINGDATA_H
