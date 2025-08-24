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
 * \file layernorm_grad_tilingdata.h
 * \brief
 */

#ifndef LIB_NORMALIZATION_LAYERNORM_GRAD_TILINGDATA_H
#define LIB_NORMALIZATION_LAYERNORM_GRAD_TILINGDATA_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LayerNormGradTiling)
    TILING_DATA_FIELD_DEF(uint32_t, stackBufferSize);
    TILING_DATA_FIELD_DEF(uint32_t, bLength);
    TILING_DATA_FIELD_DEF(uint32_t, sLength);
    TILING_DATA_FIELD_DEF(uint32_t, hLength);
    TILING_DATA_FIELD_DEF(uint32_t, originalHLength);
    TILING_DATA_FIELD_DEF(uint32_t, oneCalSize);
    TILING_DATA_FIELD_DEF(uint32_t, nohCalSize);
    TILING_DATA_FIELD_DEF(uint32_t, loopNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailSize);
    TILING_DATA_FIELD_DEF(uint32_t, nohTailSize);
    TILING_DATA_FIELD_DEF(uint32_t, tmpTensorBSHPos);
    TILING_DATA_FIELD_DEF(uint32_t, tmpTensorBSHSize);
    TILING_DATA_FIELD_DEF(uint32_t, pdVarTensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, pdVarTensorSize);
    TILING_DATA_FIELD_DEF(uint32_t, pdMeanTensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, pdMeanTensorSize);
    TILING_DATA_FIELD_DEF(uint32_t, x1TensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, x1TensorSize);
    TILING_DATA_FIELD_DEF(uint32_t, x2TensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, x2TensorSize);
    TILING_DATA_FIELD_DEF(uint32_t, x3TensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, x3TensorSize);
    TILING_DATA_FIELD_DEF(uint32_t, tmpTensorPos);
    TILING_DATA_FIELD_DEF(uint32_t, tmpTensorSize);
    TILING_DATA_FIELD_DEF(uint32_t, tmpTensor1Pos);
    TILING_DATA_FIELD_DEF(uint32_t, tmpTensor1Size);
    TILING_DATA_FIELD_DEF(uint32_t, tmpTensor2Pos);
    TILING_DATA_FIELD_DEF(uint32_t, tmpTensor2Size);
    TILING_DATA_FIELD_DEF(uint32_t, lastDimValueBack);
    TILING_DATA_FIELD_DEF(uint32_t, lastDimValueBackMulTwo);
END_TILING_DATA_DEF;
}
#endif // LIB_NORMALIZATION_LAYERNORM_GRAD_TILINGDATA_H
