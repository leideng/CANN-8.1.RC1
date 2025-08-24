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
 * \file confusion_transpose_base_impl.h
 * \brief
 */
#ifndef IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_BASE_IMPL_H
#define IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_BASE_IMPL_H
#include "confusion_transpose_base_2nd012.h"

namespace AscendC {
template <typename T> struct ConfusionTranspose012Params {
    __aicore__ ConfusionTranspose012Params(){};

    int32_t i;
    int32_t j;
    int32_t k;

    TransposeType transposeType;
    TransDataTo5HDParams transDataParams1;
    TransDataTo5HDParams transDataParams2;

    uint32_t tmp1RemainRowCount;
    uint32_t tmp2NeedRowCount;
    uint32_t transdataRepeat;
    uint32_t tmp2Count;
    uint32_t tmp1Count;
    uint32_t dstAllCount;
    uint32_t dstPreHnCount;

    // main：src->tmp1
    uint64_t dstLocalList1[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList1[NCHW_CONV_ADDR_LIST_SIZE];

    // main：tmp2->dst
    uint64_t dstLocalList2[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList2[NCHW_CONV_ADDR_LIST_SIZE];
};

template <typename T>
__aicore__ inline void InitConfusionTranspose012TransParams(ConfusionTranspose012Tiling& tiling,
    ConfusionTranspose012Params<T>& params, TransposeType transposeTypeIn)
{
    if (tiling.shapeH < 16) {
        params.tmp2NeedRowCount = tiling.shapeH;
    }
    params.transposeType = transposeTypeIn;

    params.tmp1RemainRowCount = 0; // records the number of lines stored in tmp1
    params.tmp2NeedRowCount = 16;  // tmp2 need lines
    params.transdataRepeat = 0;    // times of transposing to dst
    params.tmp2Count = 0;          // number of lines in tmp2
    params.tmp1Count = 0;          // line from which data needs to be moved in tmp1
    params.dstAllCount = 0;        // number of columns in dst
    params.dstPreHnCount = 0;      // number of columns in each H/N in dst

    params.transDataParams1.repeatTimes = 1;
    params.transDataParams1.dstRepStride = 0;
    params.transDataParams1.srcRepStride = 0;

    params.transDataParams2.repeatTimes = 1;
    params.transDataParams2.dstRepStride = 0;
    params.transDataParams2.srcRepStride = 0;
}

template <typename T>
__aicore__ inline void ConfusionTranspose012Tmp1RemainRowCountZero(const LocalTensor<T>& srcTensor,
    ConfusionTranspose012Tiling& tiling, ConfusionTranspose012Params<T>& params, const LocalTensor<T>& tmp1)
{
    // src->tmp1
    if (params.tmp1RemainRowCount != 0) {
        return;
    }
    if constexpr (sizeof(T) == sizeof(half)) {
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
            params.dstLocalList1[n] = (uint64_t)tmp1[BLOCK_CUBE * n].GetPhyAddr();
            params.srcLocalList1[n] = (uint64_t)srcTensor[params.i * tiling.alignsCube +
                params.j * CUBE_MAX_SIZE + BLOCK_CUBE * n + params.k * tiling.srcBatchOffset].GetPhyAddr();
        }
        PipeBarrier<PIPE_V>();
        TransDataTo5HD<T>(params.dstLocalList1, params.srcLocalList1, params.transDataParams1);
    } else if constexpr (sizeof(T) == sizeof(float)) {
        for (uint16_t m = 0; m < 2; m++) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                params.dstLocalList1[n] = (uint64_t)tmp1[m * CUBE_HALF_SIZE + tiling.blockSize * n].GetPhyAddr();
            }
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                params.srcLocalList1[n] =
                    (uint64_t)srcTensor[params.i * tiling.alignsCube + params.j * CUBE_MAX_SIZE +
                        m * tiling.blockSize + BLOCK_CUBE * n + params.k * tiling.srcBatchOffset].GetPhyAddr();
            }
            PipeBarrier<PIPE_V>();
            TransDataTo5HD<T>(params.dstLocalList1, params.srcLocalList1, params.transDataParams1);
        }
    }
    // update the number of rows stored in tmp1
    if (tiling.hnDiv > BLOCK_CUBE) {
        if ((params.dstPreHnCount < tiling.hnDiv) && (params.dstPreHnCount + BLOCK_CUBE) > tiling.hnDiv) {
            params.tmp1RemainRowCount = tiling.hnDiv - params.dstPreHnCount;
        } else {
            params.tmp1RemainRowCount = BLOCK_CUBE;
        }
    } else if (tiling.hnDiv <= BLOCK_CUBE) {
        params.tmp1RemainRowCount = tiling.hnDiv;
    }
}

template <typename T>
__aicore__ inline void ConfusionTranspose012Tmp1ToTmp2(const LocalTensor<T>& tmp2, ConfusionTranspose012Tiling& tiling,
    ConfusionTranspose012Params<T>& params, const LocalTensor<T>& tmp1)
{
    // tmp1->tmp2：transfer the number of valid dst lines to tmp2
    if (params.tmp2NeedRowCount <= params.tmp1RemainRowCount) {
        if (params.tmp2NeedRowCount != 0) {
            DataCopyParams dataCopyParams1;
            dataCopyParams1.blockCount = 1;
            dataCopyParams1.blockLen = params.tmp2NeedRowCount * tiling.blockNum;
            PipeBarrier<PIPE_V>();
            DataCopy(tmp2[(params.tmp2Count * BLOCK_CUBE)], tmp1, dataCopyParams1);
            params.tmp1Count += params.tmp2NeedRowCount;
            params.dstAllCount += params.tmp2NeedRowCount;
            params.dstPreHnCount += params.tmp2NeedRowCount;
            params.tmp2Count += params.tmp2NeedRowCount;
            params.tmp1RemainRowCount -= params.tmp2NeedRowCount;
            params.tmp2NeedRowCount = 0;
        }
    } else if (params.tmp2NeedRowCount > params.tmp1RemainRowCount) {
        if (params.tmp1RemainRowCount != 0) {
            DataCopyParams dataCopyParams1;
            dataCopyParams1.blockCount = 1;
            dataCopyParams1.blockLen = params.tmp1RemainRowCount * tiling.blockNum;
            PipeBarrier<PIPE_V>();
            DataCopy(tmp2[(params.tmp2Count * BLOCK_CUBE)], tmp1, dataCopyParams1);
            params.tmp1Count += params.tmp1RemainRowCount;
            params.dstAllCount += params.tmp1RemainRowCount;
            params.dstPreHnCount += params.tmp1RemainRowCount;
            params.tmp2Count += params.tmp1RemainRowCount;
            params.tmp2NeedRowCount -= params.tmp1RemainRowCount;
            params.tmp1RemainRowCount = 0;
        }
    }
}

template <typename T>
__aicore__ inline void ConfusionTranspose012Tmp2ToDstHalf(const LocalTensor<T>& dstTensor,
    ConfusionTranspose012Tiling& tiling, ConfusionTranspose012Params<T>& params, const LocalTensor<T>& tmp2)
{
    if (params.transposeType == TransposeType::TRANSPOSE_NZ2ND_012_WITHOUT_N) {
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
            params.dstLocalList2[n] = (uint64_t)dstTensor[params.transdataRepeat * BLOCK_CUBE +
                params.j * tiling.alignhBlockCube + tiling.alignH * n + params.k * tiling.dstBatchOffset].GetPhyAddr();
        }
    } else if (params.transposeType == TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N) {
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
            params.dstLocalList2[n] = (uint64_t)dstTensor[params.transdataRepeat * tiling.alignsCube +
                params.j * CUBE_MAX_SIZE + BLOCK_CUBE * n + params.k * tiling.dstBatchOffset].GetPhyAddr();
        }
    }
    for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
        params.srcLocalList2[n] = (uint64_t)tmp2[BLOCK_CUBE * n].GetPhyAddr();
    }
    PipeBarrier<PIPE_V>();
    TransDataTo5HD<T>(params.dstLocalList2, params.srcLocalList2, params.transDataParams2);
}

template <typename T>
__aicore__ inline void ConfusionTranspose012Tmp2ToDstFloat(const LocalTensor<T>& dstTensor,
    ConfusionTranspose012Tiling& tiling, ConfusionTranspose012Params<T>& params, const LocalTensor<T>& tmp2)
{
    for (uint16_t m = 0; m < 2; m++) {
        if (params.transposeType == TransposeType::TRANSPOSE_NZ2ND_012_WITHOUT_N) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n = n + 2) {
                params.dstLocalList2[n] = (uint64_t)dstTensor[params.transdataRepeat * BLOCK_CUBE +
                    params.j * tiling.alignhBlockCube + m * tiling.blockSizeMulAlignH +
                    tiling.alignH * (n / 2) + params.k * tiling.dstBatchOffset].GetPhyAddr();
                params.dstLocalList2[n + 1] = (uint64_t)dstTensor[params.transdataRepeat * BLOCK_CUBE +
                    params.j * tiling.alignhBlockCube + m * tiling.blockSizeMulAlignH + tiling.alignH * (n / 2) +
                    tiling.blockSize + params.k * tiling.dstBatchOffset].GetPhyAddr();
            }
        } else if (params.transposeType == TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                params.dstLocalList2[n] = (uint64_t)dstTensor[params.transdataRepeat * tiling.alignsCube +
                    params.j * CUBE_MAX_SIZE + m * CUBE_HALF_SIZE + tiling.blockSize * n +
                    params.k * tiling.dstBatchOffset].GetPhyAddr();
            }
        }
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
            params.srcLocalList2[n] = (uint64_t)tmp2[m * tiling.blockSize + BLOCK_CUBE * n].GetPhyAddr();
        }
        PipeBarrier<PIPE_V>();
        TransDataTo5HD<T>(params.dstLocalList2, params.srcLocalList2, params.transDataParams2);
    }
}

template <typename T>
__aicore__ inline void ConfusionTranspose012Tmp2ToDst(const LocalTensor<T>& dstTensor,
    ConfusionTranspose012Tiling& tiling, ConfusionTranspose012Params<T>& params, const LocalTensor<T>& tmp2)
{
    if (params.tmp2NeedRowCount == 0) {
        if constexpr (sizeof(T) == sizeof(half)) {
            ConfusionTranspose012Tmp2ToDstHalf(dstTensor, tiling, params, tmp2);
        } else if constexpr (sizeof(T) == sizeof(float)) {
            ConfusionTranspose012Tmp2ToDstFloat(dstTensor, tiling, params, tmp2);
        }
        // each time tmp2->dst is transposed, tmp2Count that points to the stored data in tmp2 is set to 0
        params.tmp2Count = 0;
        params.transdataRepeat += 1;
        if (params.dstAllCount == tiling.shapeH) {
            params.tmp1RemainRowCount = 0;
            params.dstAllCount = 0;
        }
        // tmp2NeedRowCount
        if ((params.transdataRepeat + 1) != tiling.hBlockNum) {
            params.tmp2NeedRowCount = BLOCK_CUBE;
        } else {
            params.tmp2NeedRowCount = tiling.shapeH - params.transdataRepeat * BLOCK_CUBE;
        }
    }
}

template <typename T>
__aicore__ inline void ConfusionTranspose012Tmp1ToTmp2Remain(const LocalTensor<T>& tmp2,
    ConfusionTranspose012Tiling& tiling, ConfusionTranspose012Params<T>& params, const LocalTensor<T>& tmp1)
{
     // tmp1->tmp2
    if (params.tmp1RemainRowCount >= params.tmp2NeedRowCount) {
        if (params.tmp2NeedRowCount != 0) {
            DataCopyParams dataCopyParams2;
            dataCopyParams2.blockCount = 1;
            dataCopyParams2.blockLen = params.tmp2NeedRowCount * tiling.blockNum;
            PipeBarrier<PIPE_V>();
            DataCopy(tmp2[(params.tmp2Count * BLOCK_CUBE)], tmp1[(params.tmp1Count * BLOCK_CUBE)], dataCopyParams2);
            params.dstAllCount += params.tmp2NeedRowCount;
            params.dstPreHnCount += params.tmp2NeedRowCount;
            params.tmp2Count += params.tmp2NeedRowCount;
            params.tmp1RemainRowCount -= params.tmp2NeedRowCount;
            params.tmp2NeedRowCount = 0;
        }
    } else {
        if (params.tmp2NeedRowCount != 0) {
            DataCopyParams dataCopyParams2;
            dataCopyParams2.blockCount = 1;
            dataCopyParams2.blockLen = params.tmp1RemainRowCount * tiling.blockNum;
            PipeBarrier<PIPE_V>();
            DataCopy(tmp2[(params.tmp2Count * BLOCK_CUBE)], tmp1[(params.tmp1Count * BLOCK_CUBE)], dataCopyParams2);
            params.dstAllCount += params.tmp1RemainRowCount;
            params.dstPreHnCount += params.tmp1RemainRowCount;
            params.tmp2Count = params.tmp1RemainRowCount;
            params.tmp2NeedRowCount -= params.tmp1RemainRowCount;
            params.tmp1RemainRowCount = 0;
        }
    }
}

/*
scene1：{ shape:[B, A1, A3 / 16, A2 / 16, 16, 16], format:"NZ"} -->{ shape:[B, A2, A1, A3], ori_shape:[B, A2, A1, A3],
format:"ND"} scene2： { shape:[B, A1, A3 / 16, A2 / 16, 16, 16], format:"NZ"}-->{ shape:[B, A2, A3 / 16, A1 / 16, 16,
16], origin_shape:[B, A2, A1, A3], format:"NZ"}
*/
template <typename T>
__aicore__ inline void ConfusionTranspose0213Compute(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, TransposeType transposeTypeIn, ConfusionTranspose0213Tiling &tiling)
{
    ConfusionTranspose0213Params<T> params;
    InitConfusionTranspose0213TransParams<T>(tiling, params, transposeTypeIn);

    LocalTensor<T> tmp1 = sharedTmpBuffer.ReinterpretCast<T>();

    for (params.k = 0; params.k < tiling.shapeB; params.k++) {
        for (params.i = 0; params.i < tiling.shapeA1; params.i++) {
            for (params.j = 0; params.j < tiling.widthTiling; params.j++) {
                for (params.m = 0; params.m < tiling.mainBlocks; params.m++) {
                    // b16
                    if constexpr (sizeof(T) == sizeof(half)) {
                        ConfusionTranspose0213MainHalf(dstTensor, srcTensor, params, tiling, tmp1);
                    } else if constexpr (sizeof(T) == sizeof(float)) {
                        ConfusionTranspose0213MainFloat(dstTensor, srcTensor, params, tiling, tmp1);
                    } else {
                        ASSERT(false);
                    }
                }
                if (tiling.tailSize) {
                    if constexpr (sizeof(T) == sizeof(half)) {
                        ConfusionTranspose0213TailHalf(dstTensor, srcTensor, params, tiling, tmp1);
                    } else if constexpr (sizeof(T) == sizeof(float)) {
                        ConfusionTranspose0213TailFloat(dstTensor, srcTensor, params, tiling, tmp1);
                    } else {
                        ASSERT(false);
                    }
                }
            }
        }
    }
}

/*
scene3：{ shape:[B, H / 16, S / 16, 16, 16], format:"NZ"}-->{ shape:[B, N, H/N/16, S / 16, 16, 16], ori_shape:[B, N, S,
H/N], format:"NZ"}
*/
template <typename T>
__aicore__ inline void ConfusionTranspose2NZ012NCompute(const LocalTensor<T> &dstTensor,
    const LocalTensor<T> &srcTensor, const LocalTensor<uint8_t> &sharedTmpBuffer,
    ConfusionTranspose2NZ012NTiling &tiling)
{
    LocalTensor<T> tmp1 = sharedTmpBuffer.ReinterpretCast<T>();
    LocalTensor<T> tmp2 = tmp1[CUBE_MAX_SIZE];

    ConfusionTranspose2NZ012NParams<T> params;
    InitConfusionTranspose2NZ012N(params, tiling);

    for (params.k = 0; params.k < tiling.shapeB; params.k++) {
        params.transdataRepeat = 0;
        for (params.j = 0; params.j < tiling.sBlockNum; params.j++) {
            for (params.i = 0; params.i < tiling.hBlockNum; params.i++) {
                ConfusionTranspose2NZ012NTmp1RemainRowCountZero(srcTensor, tiling, params, tmp1);
                ConfusionTranspose2NZ012NTmp2RemainRowCountFirst(tiling, params, tmp1, tmp2);
                ConfusionTranspose2NZ012NTmp2RemainRowCountZero(dstTensor, tiling, params, tmp2);
                ConfusionTranspose2NZ012NCalcTmp2NeedRowCount(tiling, params);
                // When tmp1RemainRowCount has more cached rows, the system continues to process tmp1->tmp2 and
                // tmp2->dst.
                while (params.tmp1RemainRowCount) {
                    ConfusionTranspose2NZ012NTmp2NeedRowCount(tiling, params, tmp1, tmp2);
                    ConfusionTranspose2NZ012NTmp2NeedRowCountZero(dstTensor, tiling, params, tmp2);
                }
            }
        }
    }
}

/*
scene4：{ shape:[B, H / 16, S / 16, 16, 16], format:"NZ"}-->{ shape:[B, N, S, H/N], ori_shape:[B, N, S, H/N],
format:"ND"}
*/
template <typename T>
__aicore__ inline void ConfusionTranspose2ND012NCompute(const LocalTensor<T> &dstTensor,
    const LocalTensor<T> &srcTensor, const LocalTensor<uint8_t> &sharedTmpBuffer,
    ConfusionTranspose2ND012NTiling &tiling)
{
    LocalTensor<T> tmp1 = sharedTmpBuffer.ReinterpretCast<T>();
    LocalTensor<T> tmp2 = tmp1[CUBE_MAX_SIZE];

    ConfusionTranspose2ND012NParams<T> params;
    InitConfusionTranspose2ND012N(params, tiling);

    for (params.k = 0; params.k < tiling.shapeB; params.k++) {
        params.transdataRepeat = 0;
        for (params.j = 0; params.j < tiling.sBlockNum; params.j++) {
            for (params.i = 0; params.i < tiling.hBlockNum; params.i++) {
                ConfusionTranspose2ND012NTmp1RemainRowCount(srcTensor, tiling, params, tmp1);
                ConfusionTranspose2ND012NTmp1RemainRowCountFirst(tiling, params, tmp1, tmp2);
                ConfusionTranspose2ND012NTmp2NeedRowCount(dstTensor, tiling, params, tmp2);
                // When tmp1RemainRowCount has more cached rows, the system continues to process tmp1->tmp2 and
                // tmp2->dst.
                while (params.tmp1RemainRowCount) {
                    ConfusionTranspose2ND012NTmp2NeedRowCountNotZero(tiling, params, tmp1, tmp2);
                    ConfusionTranspose2ND012Ntmp2NeedRowCountZero(dstTensor, tiling, params, tmp2);
                }
            }
        }
    }
}

/*
scene5：{ shape:[B, N, H/N/16, S/16, 16, 16], format:"NZ"}-->{ shape:[B, S, H], ori_shape:[B, S, H], format:"ND"}
scene6：{ shape:[B, N, H/N/16, S/16, 16, 16], format:"NZ"}-->{ shape:[B, H/16, S/16, 16, 16], ori_shape:[B, S, H],
format:"NZ"}
*/
template <typename T>
__aicore__ inline void ConfusionTranspose012Compute(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, TransposeType transposeTypeIn, ConfusionTranspose012Tiling &tiling)
{
    ConfusionTranspose012Params<T> params;
    InitConfusionTranspose012TransParams<T>(tiling, params, transposeTypeIn);
    LocalTensor<T> tmp1 = sharedTmpBuffer.ReinterpretCast<T>();
    LocalTensor<T> tmp2 = tmp1[CUBE_MAX_SIZE];

    for (params.k = 0; params.k < tiling.shapeB; params.k++) {
        params.transdataRepeat = 0;
        for (params.j = 0; params.j < tiling.sBlockNum; params.j++) {
            params.transdataRepeat = 0;
            for (params.i = 0; params.i < (tiling.hnDivBlockNum * tiling.shapeN); params.i++) {
                // update tmp1Count each time the fractal is moved from src to tmp1
                params.tmp1Count = 0;
                ConfusionTranspose012Tmp1RemainRowCountZero(srcTensor, tiling, params, tmp1);
                // update dstPreHnCount
                if (params.dstPreHnCount == tiling.hnDiv) {
                    params.dstPreHnCount = 0;
                }
                ConfusionTranspose012Tmp1ToTmp2(tmp2, tiling, params, tmp1);
                ConfusionTranspose012Tmp2ToDst(dstTensor, tiling, params, tmp2);
                // when tmp1RemainRowCount has more cached rows, the system continues to process tmp1->tmp2 and
                // tmp2->dst.
                while (params.tmp1RemainRowCount) {
                    // tmp1->tmp2
                    ConfusionTranspose012Tmp1ToTmp2Remain(tmp2, tiling, params, tmp1);
                    // tmp2->dst
                    ConfusionTranspose012Tmp2ToDst(dstTensor, tiling, params, tmp2);
                }
            }
        }
    }
}

/*
scene7：{ shape:[H, W], format:"ND"} -->{ shape:[W, H], format:"ND"}
*/
template <typename T>
__aicore__ inline void ConfusionTransposeOnlyCompute(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    ConfusionTransposeOnlyTiling &tiling)
{
    uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = tiling.repeat;
    transDataParams.dstRepStride = transDataParams.repeatTimes > 1 ? tiling.stride : 0;
    transDataParams.srcRepStride = transDataParams.repeatTimes > 1 ? 1 : 0;
    for (int32_t i = 0; i < tiling.highBlock; i++) {
        if constexpr (sizeof(T) == sizeof(half)) {
            for (int32_t m = 0; m < NCHW_CONV_ADDR_LIST_SIZE; m++) {
                dstLocalList[m] = (uint64_t)dstTensor[i * BLOCK_CUBE + tiling.height * m].GetPhyAddr();
            }
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                srcLocalList[n] =
                    (uint64_t)srcTensor[i * tiling.width * BLOCK_CUBE + tiling.width * n].GetPhyAddr();
            }
            TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        } else if constexpr (sizeof(T) == sizeof(float)) {
            for (int32_t m = 0; m < NCHW_CONV_ADDR_LIST_SIZE; m = m + 2) {
                dstLocalList[m] = (uint64_t)dstTensor[i * BLOCK_CUBE + tiling.height * (m / 2)].GetPhyAddr();
                dstLocalList[m + 1] =
                    (uint64_t)dstTensor[i * BLOCK_CUBE + tiling.height * (m / 2) + tiling.blockSize].GetPhyAddr();
            }
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                srcLocalList[n] =
                    (uint64_t)srcTensor[i * tiling.width * BLOCK_CUBE + tiling.width * n].GetPhyAddr();
            }
            TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        }
    }
}
} // namespace AscendC
#endif // IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_BASE_IMPL_H