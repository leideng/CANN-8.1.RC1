/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file data_copy_transpose_grad.h
 * \brief
 */
#ifndef DATA_COPY_TRANSPOSE_H
#define DATA_COPY_TRANSPOSE_H

#include "kernel_operator.h"

using namespace AscendC;

enum class DataCopyTransposeType {
    TRANSPOSE_TYPE_NONE,  // 默认值
    TRANSPOSE_NZ2ND_0213, // { shape:[B, A1, A3 / 16, A2 / 16, 16, 16], format:"NZ"} -->{ shape:[B, A2, A1, A3],
                          // ori_shape:[B, A2, A1, A3], format:"ND"}
    TRANSPOSE_NZ2NZ_0213, // { shape:[B, A1, A3 / 16, A2 / 16, 16, 16], format:"NZ"}-->{ shape:[B, A2, A3 / 16, A1 / 16,
                          // 16, 16], origin_shape:[B, A2, A1, A3], format:"NZ"}
    TRANSPOSE_NZ2NZ_012_WITH_N,    // { shape:[B, H / 16, S / 16, 16, 16], format:"NZ"}-->{ shape:[B, N, H/N/16, S / 16,
                                   // 16, 16], ori_shape:[B, N, S, H/N], format:"NZ"}
    TRANSPOSE_NZ2ND_012_WITH_N,    // { shape:[B, H / 16, S / 16, 16, 16], format:"NZ"}-->{ shape:[B, N, S, H/N],
                                   // ori_shape:[B, N, S, H/N], format:"ND"}
    TRANSPOSE_NZ2ND_012_WITHOUT_N, // { shape:[B, N, H/N/16, S/16, 16, 16], format:"NZ"}-->{ shape:[B, S, H],
                                   // ori_shape:[B, S, H], format:"ND"}
    TRANSPOSE_NZ2NZ_012_WITHOUT_N, // { shape:[B, N, H/N/16, S/16, 16, 16], format:"NZ"}-->{ shape:[B, H/16, S/16, 16,
                                   // 16], ori_shape:[B, S, H], format:"NZ"}
    TRANSPOSE_ND2ND_ONLY,          // { shape:[H, W], format:"ND"} -->{ shape:[W, H], format:"ND"}
    TRANSPOSE_ND_UB_GM,            //  [B, N, S, H/N] -> [B, S, H]
};

__aicore__ inline void DataCopyUB2GMAlign(const GlobalTensor<half> &dstGlobal, const LocalTensor<half> &srcLocal,
    uint16_t nBurst, uint32_t lenBurst, uint8_t srcGap, uint8_t dstGap)
{
    copy_ubuf_to_gm_align_b16((__gm__ void *)dstGlobal.GetPhyAddr(), (__ubuf__ void *)srcLocal.GetPhyAddr(), 0, nBurst,
        lenBurst, 0, 0, srcGap, dstGap);
}

__aicore__ inline void DataCopyUB2GMAlign(const GlobalTensor<float> &dstGlobal, const LocalTensor<float> &srcLocal,
    uint16_t nBurst, uint32_t lenBurst, uint8_t srcGap, uint8_t dstGap)
{
    copy_ubuf_to_gm_align_b32((__gm__ void *)dstGlobal.GetPhyAddr(), (__ubuf__ void *)srcLocal.GetPhyAddr(), 0, nBurst,
        lenBurst, 0, 0, srcGap, dstGap);
}

// 切分后的小块在原有大块里的位置索引
struct TransposeParams {
    int bIndex;
    int nIndex;
    int sIndex;
    int hNIndex;
};

template <typename T>
__aicore__ inline void GetDataCopyTransposeTiling(const GlobalTensor<T> &dstGlobal, const LocalTensor<T> &srcLocal,
    TransposeParams transposeParams, CopyTransposeTiling& tiling)
{
    ShapeInfo srcShapeInfo = srcLocal.GetShapeInfo();
    ShapeInfo dstShapeInfo = dstGlobal.GetShapeInfo();
    tiling.dstShapeB = dstShapeInfo.originalShape[0];
    tiling.dstShapeN = dstShapeInfo.originalShape[1];
    tiling.dstShapeS = dstShapeInfo.originalShape[2];
    tiling.dstShapeHN = dstShapeInfo.originalShape[3];
    tiling.dstShapeH = dstShapeInfo.shape[2];

    tiling.srcShapeB = srcShapeInfo.shape[0];
    tiling.srcShapeN = srcShapeInfo.shape[1];
    tiling.srcShapeS = srcShapeInfo.shape[2];
    tiling.srcShapeHN = srcShapeInfo.shape[3];
    tiling.originalShapeNLen = srcShapeInfo.originalShape[3] * sizeof(T);
    tiling.shapeSHValue = tiling.dstShapeS * tiling.dstShapeH;
    tiling.shapeNsValue = tiling.srcShapeHN * tiling.srcShapeS;
    tiling.shapeNsnValue = tiling.srcShapeHN * tiling.srcShapeS * tiling.srcShapeN;
}

template <typename T>
__aicore__ inline void DataCopyTranspose(const GlobalTensor<T> &dstGlobal, const LocalTensor<T> &srcLocal,
    DataCopyTransposeType transposeType, TransposeParams transposeParams, CopyTransposeTiling tiling)
{
    if (transposeType != DataCopyTransposeType::TRANSPOSE_ND_UB_GM) {
        return;
    }
    if (tiling.dstShapeB == 0) {
        GetDataCopyTransposeTiling(dstGlobal, srcLocal, transposeParams, tiling);
    }

    int startAddr = transposeParams.bIndex * (tiling.shapeSHValue) + transposeParams.nIndex * tiling.dstShapeHN +
        transposeParams.sIndex * tiling.dstShapeH + transposeParams.hNIndex;

    for (int i = 0; i < tiling.srcShapeB; i++) {
        for (int j = 0; j < tiling.srcShapeN; j++) {
            for (int k = 0; k < tiling.srcShapeS; k++) {
                DataCopyUB2GMAlign(
                    dstGlobal[startAddr + i * (tiling.shapeSHValue) + j * tiling.dstShapeHN + k * tiling.dstShapeH],
                    srcLocal[k * tiling.srcShapeHN + j * (tiling.shapeNsValue) + i * (tiling.shapeNsnValue)], 1,
                    tiling.originalShapeNLen, 0, 0);
            }
        }
    }
}
#endif // DATA_COPY_TRANSPOSE_H