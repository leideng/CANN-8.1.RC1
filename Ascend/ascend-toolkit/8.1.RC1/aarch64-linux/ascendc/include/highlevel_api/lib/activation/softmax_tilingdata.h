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
 * \file softmax_tilingdata.h
 * \brief
 */

#ifndef LIB_SOFTMAX_SOFTMAX_TILINGDATA_H
#define LIB_SOFTMAX_SOFTMAX_TILINGDATA_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftMaxTiling)
    TILING_DATA_FIELD_DEF(uint32_t, srcM);
    TILING_DATA_FIELD_DEF(uint32_t, srcK);
    TILING_DATA_FIELD_DEF(uint32_t, srcSize);
    TILING_DATA_FIELD_DEF(uint32_t, outMaxM);
    TILING_DATA_FIELD_DEF(uint32_t, outMaxK);
    TILING_DATA_FIELD_DEF(uint32_t, outMaxSize);
    TILING_DATA_FIELD_DEF(uint32_t, splitM);
    TILING_DATA_FIELD_DEF(uint32_t, splitK);
    TILING_DATA_FIELD_DEF(uint32_t, splitSize);
    TILING_DATA_FIELD_DEF(uint32_t, reduceM);
    TILING_DATA_FIELD_DEF(uint32_t, reduceK);
    TILING_DATA_FIELD_DEF(uint32_t, reduceSize);
    TILING_DATA_FIELD_DEF(uint32_t, rangeM);
    TILING_DATA_FIELD_DEF(uint32_t, tailM);
    TILING_DATA_FIELD_DEF(uint32_t, tailSplitSize);
    TILING_DATA_FIELD_DEF(uint32_t, tailReduceSize);
END_TILING_DATA_DEF;
}
#endif // LIB_SOFTMAX_SOFTMAX_TILINGDATA_H