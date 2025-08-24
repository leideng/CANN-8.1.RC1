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
 * \file pad_v220_impl.h
 * \brief
 */
#ifndef IMPL_PAD_PAD_PAD_V220_IMPL_H
#define IMPL_PAD_PAD_PAD_V220_IMPL_H

#include "pad_base_impl.h"

namespace AscendC {
template <typename T>
__aicore__ inline void PadCompute(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    PadParams &padParams, const LocalTensor<uint8_t> &sharedTmpBuffer, PadTiling &tiling)
{
    uint32_t width = tiling.srcWidth;

    // 32B aligned
    if (width * sizeof(T) % ONE_BLK_SIZE == 0) {
        AlignedPad(dstTensor, srcTensor, padParams, tiling);
    } else {
        LocalTensor<T> tmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();
        UnAlignedPad(dstTensor, srcTensor, padParams, tmpBuffer, tiling);
    }
}
/* **************************************************************************************************
 * UnPad                                             *
 * ************************************************************************************************* */
/*
 * @ingroup UnPad
 * @brief unpad from src to dst, applicable to vector data
 * @param [out] dstTensor output LocalTensor
 * @param [in] srcTensor input LocalTensor
 * @param [in] sharedTmpBuffer tmp buffer LocalTensor
 * @param [in] unPadParams.leftPad number of left unpad
 * @param [in] unPadParams.rightPad number of right unpad
 */
template <typename T>
__aicore__ inline void UnPadCompute(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    UnPadParams &unPadParams, LocalTensor<uint8_t> &sharedTmpBuffer, UnPadTiling &tiling)
{
    uint16_t rightPad = unPadParams.rightPad;
    uint16_t height = tiling.srcHeight;
    uint16_t width = tiling.srcWidth;

    GatherMaskParams reducev2Params;
    reducev2Params.repeatTimes = height;
    reducev2Params.src0RepeatStride = static_cast<uint16_t>(width * sizeof(T) / ONE_BLK_SIZE);
    uint64_t rsvdCnt = 0;
    GatherMask(dstTensor, srcTensor, REDUCEV2_MODE_SEVEN, true, (width - rightPad), reducev2Params, rsvdCnt);
    ResetMask();
}
} // namespace AscendC
#endif // IMPL_PAD_PAD_PAD_V220_IMPL_H
