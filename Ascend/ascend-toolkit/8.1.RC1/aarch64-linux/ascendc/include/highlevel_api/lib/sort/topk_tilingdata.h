/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LIB_SORT_TOPK_TILINGDATA_H
#define LIB_SORT_TOPK_TILINGDATA_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TopkTiling)
    TILING_DATA_FIELD_DEF(int32_t, tmpLocalSize);
    TILING_DATA_FIELD_DEF(int32_t, allDataSize);
    TILING_DATA_FIELD_DEF(int32_t, innerDataSize);
    TILING_DATA_FIELD_DEF(uint32_t, sortRepeat);
    TILING_DATA_FIELD_DEF(int32_t, mrgSortRepeat);
    TILING_DATA_FIELD_DEF(int32_t, kAlignFourBytes);
    TILING_DATA_FIELD_DEF(int32_t, kAlignTwoBytes);
    TILING_DATA_FIELD_DEF(int32_t, maskOffset);
    TILING_DATA_FIELD_DEF(int32_t, maskVreducev2FourBytes);
    TILING_DATA_FIELD_DEF(int32_t, maskVreducev2TwoBytes);
    TILING_DATA_FIELD_DEF(int32_t, mrgSortSrc1offset);
    TILING_DATA_FIELD_DEF(int32_t, mrgSortSrc2offset);
    TILING_DATA_FIELD_DEF(int32_t, mrgSortSrc3offset);
    TILING_DATA_FIELD_DEF(int32_t, mrgSortTwoQueueSrc1Offset);
    TILING_DATA_FIELD_DEF(int32_t, mrgFourQueueTailPara1);
    TILING_DATA_FIELD_DEF(int32_t, mrgFourQueueTailPara2);
    TILING_DATA_FIELD_DEF(int32_t, srcIndexOffset);
    TILING_DATA_FIELD_DEF(uint32_t, copyUbToUbBlockCount);
    TILING_DATA_FIELD_DEF(int32_t, topkMrgSrc1MaskSizeOffset);
    TILING_DATA_FIELD_DEF(int32_t, topkNSmallSrcIndexOffset);
    TILING_DATA_FIELD_DEF(uint32_t, vreduceValMask0);
    TILING_DATA_FIELD_DEF(uint32_t, vreduceValMask1);
    TILING_DATA_FIELD_DEF(uint32_t, vreduceIdxMask0);
    TILING_DATA_FIELD_DEF(uint32_t, vreduceIdxMask1);
    TILING_DATA_FIELD_DEF(uint16_t, vreducehalfValMask0);
    TILING_DATA_FIELD_DEF(uint16_t, vreducehalfValMask1);
    TILING_DATA_FIELD_DEF(uint16_t, vreducehalfValMask2);
    TILING_DATA_FIELD_DEF(uint16_t, vreducehalfValMask3);
    TILING_DATA_FIELD_DEF(uint16_t, vreducehalfValMask4);
    TILING_DATA_FIELD_DEF(uint16_t, vreducehalfValMask5);
    TILING_DATA_FIELD_DEF(uint16_t, vreducehalfValMask6);
    TILING_DATA_FIELD_DEF(uint16_t, vreducehalfValMask7);
END_TILING_DATA_DEF;
}
#endif // LIB_SORT_TOPK_TILINGDATA_H