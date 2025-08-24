/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef IMPL_SELECT_SELECT_WITH_BYTES_MASK_COMMON_IMPL_H
#define IMPL_SELECT_SELECT_WITH_BYTES_MASK_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_operator_vec_cmpsel_intf.h"
#include "kernel_operator_vec_vconv_intf.h"

namespace AscendC {
// Initiliaze cmpmask before select instr since the scalar value is never changed.
template <typename T> __aicore__ inline void InitScalarSelectMask(const LocalTensor<T> &tmpMask, T scalar)
{
    SetVectorMask<half, MaskMode::COUNTER>(0, ONE_REPEAT_BYTE_SIZE / sizeof(T));
    Duplicate<T, false>(tmpMask, static_cast<T>(scalar), MASK_PLACEHOLDER, 1, 1, 8);
    PipeBarrier<PIPE_V>();

    SetCmpMask(tmpMask);
    PipeBarrier<PIPE_V>();
}

template <typename U>
__aicore__ inline void CastMaskToHalfImpl(const LocalTensor<half> &localMaskTmp, const LocalTensor<U> &mask)
{
    if constexpr (sizeof(U) == 4) {
        LocalTensor<float> tmpTensor = mask.template ReinterpretCast<float>();
        Cast<half, float, false>(localMaskTmp, tmpTensor, RoundMode::CAST_ODD, MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE / 2, DEFAULT_REPEAT_STRIDE });
    } else if constexpr (sizeof(U) == 2) {
        LocalTensor<int16_t> tmpTensor = mask.template ReinterpretCast<int16_t>();
        Cast<half, int16_t, false>(localMaskTmp, tmpTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    } else if constexpr (sizeof(U) == 1) {
        LocalTensor<uint8_t> tmpTensor = mask.template ReinterpretCast<uint8_t>();
        Cast<half, uint8_t, false>(localMaskTmp, tmpTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / 2 });
    }
}

template <typename T, typename U, bool reverse = false>
__aicore__ inline void SelectWithBytesMaskPerAxisImpl(const LocalTensor<T> &dst, const LocalTensor<T> &src0, T src1,
    const LocalTensor<U> &mask, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t srcAxisLen,
    const uint32_t bucketSize)
{
    const auto paddingLen = AlignUp(bucketSize / ONE_BYTE_BIT_SIZE, ONE_BLK_SIZE);
    const auto localMaskTmpOffset = paddingLen;
    const auto localScalarOffset = sizeof(half) * bucketSize;
    LocalTensor<uint8_t> localMask = sharedTmpBuffer;
    LocalTensor<half> localMaskTmp = sharedTmpBuffer.ReinterpretCast<half>();
    SetVectorMask<half, MaskMode::COUNTER>(0, srcAxisLen);
    CastMaskToHalfImpl<U>(localMaskTmp, mask);
    PipeBarrier<PIPE_V>();

    BinaryRepeatParams binaryParams;
    UnaryRepeatParams unaryParams;
    constexpr auto loopSize = ONE_REPEAT_BYTE_SIZE / sizeof(half);
    const auto repeatTime = DivCeil(srcAxisLen, loopSize);

    if constexpr (!reverse) {
        CompareScalar<half, uint8_t, false>(localMask, localMaskTmp, static_cast<half>(0), CMPMODE::EQ,
            MASK_PLACEHOLDER, repeatTime, unaryParams);
    } else {
        CompareScalar<half, uint8_t, false>(localMask, localMaskTmp, static_cast<half>(0), CMPMODE::NE,
            MASK_PLACEHOLDER, repeatTime, unaryParams);
    }
    PipeBarrier<PIPE_V>();
    Select(dst, localMask, src0, 1, binaryParams);
}

// When MaskLastAxis > SrcLastAxis, removes the redundant masks to make src and mask align.
template <typename U>
__aicore__ inline void RemoveRedundantMask(const LocalTensor<U> &dst, const LocalTensor<U> &mask,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const SelectWithBytesMaskShapeInfo &info)
{
    LocalTensor<uint16_t> tmpDst = dst.template ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> tmpMask = mask.template ReinterpretCast<uint16_t>();

    uint64_t rsvdCnt;

    GatherMask<uint16_t>(tmpDst, tmpMask, REDUCEV2_MODE_SEVEN, true, info.srcLastAxis * sizeof(U) / sizeof(uint16_t),
        { DEFAULT_BLK_STRIDE, static_cast<uint16_t>(info.firstAxis),
        static_cast<uint16_t>(info.maskLastAxis * sizeof(U) / ONE_BLK_SIZE), 0 },
        rsvdCnt);
    SetMaskCount();
}

template <typename T, typename U, bool reverse = false>
__aicore__ inline void SelectWithBytesMaskLoopImpl(const LocalTensor<T> &dst, const LocalTensor<T> &src0, T src1,
    const LocalTensor<U> &mask, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t loopSize,
    const uint32_t totalLen, const uint32_t srcOriginOffset, const uint32_t maskOriginOffset)
{
    for (uint32_t offset = 0; offset < totalLen; offset += loopSize) {
        auto calSize = offset + loopSize > totalLen ? totalLen - offset : loopSize;
        SelectWithBytesMaskPerAxisImpl<T, U, reverse>(dst[srcOriginOffset + offset], src0[srcOriginOffset + offset],
            src1, mask[maskOriginOffset + offset], sharedTmpBuffer, calSize, loopSize);
        PipeBarrier<PIPE_V>();
    }
}
} // namespace AscendC
#endif // IMPL_SELECT_SELECT_WITH_BYTES_MASK_COMMON_IMPL_H
