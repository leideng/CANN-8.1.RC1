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
 * \file confusion_transpose_common_impl.h
 * \brief
 */
#ifndef IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_COMMON_IMPL_H
#define IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_COMMON_IMPL_H

#if __CCE_AICORE__ <= 200
#include "confusion_transpose_v200_impl.h"
#elif __CCE_AICORE__ == 220
#include "confusion_transpose_v220_impl.h"
#endif

namespace AscendC {
template <typename T>
__aicore__ inline void ConfusionTransposeImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, TransposeType transposeType, ConfusionTransposeTiling& tiling)
{
    /*
    scene 1:{ shape:[B, A1, A3 / 16, A2 / 16, 16, 16], format:"NZ"} -->{ shape:[B, A2, A1, A3], ori_shape:[B, A2, A1,
    A3], format:"ND"} scene 2： { shape:[B, A1, A3 / 16, A2 / 16, 16, 16], format:"NZ"}-->{ shape:[B, A2, A3 / 16, A1 /
    16, 16, 16], origin_shape:[B, A2, A1, A3], format:"NZ"}
    */
    if (transposeType == TransposeType::TRANSPOSE_NZ2ND_0213 || transposeType == TransposeType::TRANSPOSE_NZ2NZ_0213) {
        ConfusionTranspose0213(dstTensor, srcTensor, sharedTmpBuffer, transposeType,
            reinterpret_cast<ConfusionTranspose0213Tiling&>(tiling));
    }
    /*
    scene 3：{ shape:[B, H / 16, S / 16, 16, 16], format:"NZ"}-->{ shape:[B, N, H/N/16, S / 16, 16, 16], ori_shape:[B, N,
    S, H/N], format:"NZ"}
    */
    else if (transposeType == TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N) {
        ConfusionTranspose2NZ012N(dstTensor, srcTensor, sharedTmpBuffer,
            reinterpret_cast<ConfusionTranspose2NZ012NTiling &>(tiling));
    }
    /*
    scene 4：{ shape:[B, H / 16, S / 16, 16, 16], format:"NZ"}-->{ shape:[B, N, S, H/N], ori_shape:[B, N, S, H/N],
    format:"ND"}
    */
    else if (transposeType == TransposeType::TRANSPOSE_NZ2ND_012_WITH_N) {
        ConfusionTranspose2ND012N(dstTensor, srcTensor, sharedTmpBuffer,
            reinterpret_cast<ConfusionTranspose2ND012NTiling &>(tiling));
    }
    /*
    scene 5：{ shape:[B, N, H/N/16, S/16, 16, 16], format:"NZ"}-->{ shape:[B, S, H], ori_shape:[B, S, H], format:"ND"}
    scene 6：{ shape:[B, N, H/N/16, S/16, 16, 16], format:"NZ"}-->{ shape:[B, H/16, S/16, 16, 16], ori_shape:[B, S, H],
    format:"NZ"}
    */
    else if (transposeType == TransposeType::TRANSPOSE_NZ2ND_012_WITHOUT_N ||
        transposeType == TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N) {
        ConfusionTranspose012(dstTensor, srcTensor, sharedTmpBuffer, transposeType,
            reinterpret_cast<ConfusionTranspose012Tiling &>(tiling));
    }
    /*
    scene 7：{ shape:[H, W], format:"ND"} -->{ shape:[W, H], format:"ND"}
    */
    else if (transposeType == TransposeType::TRANSPOSE_ND2ND_ONLY) {
        ConfusionTransposeOnly(dstTensor, srcTensor, reinterpret_cast<ConfusionTransposeOnlyTiling &>(tiling));
    }
}
} // namespace AscendC
#endif // IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_COMMON_IMPL_H