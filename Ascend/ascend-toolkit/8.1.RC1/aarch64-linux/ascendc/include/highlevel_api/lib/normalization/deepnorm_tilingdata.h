/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIB_NORMALIZATION_DEEPNORM_TILINGDATA_H
#define LIB_NORMALIZATION_DEEPNORM_TILINGDATA_H
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DeepNormTiling)
    TILING_DATA_FIELD_DEF(uint32_t, bLength);
    TILING_DATA_FIELD_DEF(uint32_t, sLength);
    TILING_DATA_FIELD_DEF(uint32_t, hLength);                   // H after alignment
    TILING_DATA_FIELD_DEF(uint32_t, originalHLength);           // H before alignment

    TILING_DATA_FIELD_DEF(uint32_t, inputXSize);                // B * S * H
    TILING_DATA_FIELD_DEF(uint32_t, meanVarSize);               // B * S
    TILING_DATA_FIELD_DEF(uint32_t, numberOfTmpBuf);            // number of B*S*H tmpBuffer needed
    TILING_DATA_FIELD_DEF(uint32_t, meanTmpTensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, meanTmpTensorSize);
    TILING_DATA_FIELD_DEF(uint32_t, varianceTmpTensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, varianceTmpTensorSize);

    TILING_DATA_FIELD_DEF(uint32_t, tmpBufSize);                // num of FP32 stored in tmpBuffer
    TILING_DATA_FIELD_DEF(uint32_t, oneTmpSize);                // after subtract mean + var, FP32 num in each tmpTensor
    TILING_DATA_FIELD_DEF(uint32_t, firstTmpStartPos);          // after tmp mean/var, first tmpTensor start position
    TILING_DATA_FIELD_DEF(uint32_t, secondTmpStartPos);         // firstTmpStartPos + oneTmpSize
    TILING_DATA_FIELD_DEF(uint32_t, thirdTmpStartPos);          // secondTmpStartPos + oneTmpSize
    TILING_DATA_FIELD_DEF(uint32_t, loopRound);                 // loops needed for tmpBuffer read all input
    TILING_DATA_FIELD_DEF(uint32_t, inputRoundSize);            // size input can read from tmpBuffer in one round
    TILING_DATA_FIELD_DEF(uint32_t, inputTailSize);
    TILING_DATA_FIELD_DEF(uint32_t, inputTailPos);
    TILING_DATA_FIELD_DEF(uint32_t, meanVarRoundSize);
    TILING_DATA_FIELD_DEF(uint32_t, meanVarTailSize);
    TILING_DATA_FIELD_DEF(uint32_t, meanVarTailPos);
    TILING_DATA_FIELD_DEF(uint32_t, bshCurLength);
    TILING_DATA_FIELD_DEF(uint32_t, bsCurLength);
    TILING_DATA_FIELD_DEF(float, lastDimValueBack);
END_TILING_DATA_DEF;
}

#endif // LIB_NORMALIZATION_DEEPNORM_TILINGDATA_H