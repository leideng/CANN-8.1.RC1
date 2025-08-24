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
 * \file pad_tilingdata.h
 * \brief
 */

#ifndef LIB_PAD_PAD_TILINGDATA_H
#define LIB_PAD_PAD_TILINGDATA_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UnPadTiling)
    TILING_DATA_FIELD_DEF(uint32_t, srcHeight);
    TILING_DATA_FIELD_DEF(uint32_t, srcWidth);
    TILING_DATA_FIELD_DEF(uint32_t, tmpBuffer1BlockNum);
    TILING_DATA_FIELD_DEF(uint32_t, tmpBuffer1RowNum);
    TILING_DATA_FIELD_DEF(uint32_t, tmpBuffer2Offset);
    TILING_DATA_FIELD_DEF(uint32_t, widthTiling);
    TILING_DATA_FIELD_DEF(uint32_t, widthFractal);
    TILING_DATA_FIELD_DEF(uint32_t, widthFractalTail);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(PadTiling)
    // common
    TILING_DATA_FIELD_DEF(uint32_t, srcHeight);
    TILING_DATA_FIELD_DEF(uint32_t, srcWidth);
    TILING_DATA_FIELD_DEF(uint32_t, srcOriWidth);
    // width 32B aligned
    TILING_DATA_FIELD_DEF(uint32_t, widthWithoutLastBlock);
    TILING_DATA_FIELD_DEF(uint32_t, blocksPerRow);
    TILING_DATA_FIELD_DEF(uint32_t, heightTiling);
    TILING_DATA_FIELD_DEF(uint32_t, heightFractal);
    TILING_DATA_FIELD_DEF(uint32_t, heightFractalTail);
    TILING_DATA_FIELD_DEF(uint32_t, mainLoopOffset);
    TILING_DATA_FIELD_DEF(uint32_t, tailBlockOffset);
    // width 32B unaligned
    TILING_DATA_FIELD_DEF(uint32_t, tmpBuffer1BlockNum);
    TILING_DATA_FIELD_DEF(uint32_t, tmpBuffer1RowNum);
    TILING_DATA_FIELD_DEF(uint32_t, tmpBuffer2Offset);
    TILING_DATA_FIELD_DEF(uint32_t, widthTiling);
    TILING_DATA_FIELD_DEF(uint32_t, widthFractal);
    TILING_DATA_FIELD_DEF(uint32_t, widthFractalTail);
    TILING_DATA_FIELD_DEF(uint32_t, widthFractalTailAlingned);
    TILING_DATA_FIELD_DEF(uint32_t, brcbTiling);
    TILING_DATA_FIELD_DEF(uint32_t, brcbFractal);
    TILING_DATA_FIELD_DEF(uint32_t, brcbFractalTail);
    TILING_DATA_FIELD_DEF(uint32_t, maxRepeatTimes);
    TILING_DATA_FIELD_DEF(uint32_t, brcbTilingRepeatTimes);
    TILING_DATA_FIELD_DEF(uint32_t, brcbTilingRepeatTimesTail);
    TILING_DATA_FIELD_DEF(uint32_t, brcbFractalTailRepeatTimes);
    TILING_DATA_FIELD_DEF(uint32_t, brcbFractalTailRepeatTimesTail);
    TILING_DATA_FIELD_DEF(uint32_t, reserved);

END_TILING_DATA_DEF;
}

#endif // LIB_PAD_PAD_TILINGDATA_H