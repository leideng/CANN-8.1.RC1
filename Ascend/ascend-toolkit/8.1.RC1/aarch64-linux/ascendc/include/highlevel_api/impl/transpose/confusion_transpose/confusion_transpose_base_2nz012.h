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
#ifndef IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_BASE_2NZ012_H
#define IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_BASE_2NZ012_H
#include "confusion_transpose_base_0213.h"

namespace AscendC {
template <typename T> struct ConfusionTranspose2NZ012NParams {
    __aicore__ ConfusionTranspose2NZ012NParams(){};

    int32_t i;
    int32_t j;
    int32_t k;

    uint32_t tmp1RemainRowCount;
    uint32_t tmp2Count;
    uint32_t tmp2NeedRowCount;
    uint32_t transdataRepeat;
    uint32_t dstPrehnCount;
    uint32_t dstAllCount;

    TransposeType transposeType;

    // main：src->tmp1
    uint64_t dstLocalList1[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList1[NCHW_CONV_ADDR_LIST_SIZE];

    // main：tmp2->dst
    uint64_t dstLocalList2[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList2[NCHW_CONV_ADDR_LIST_SIZE];

    uint64_t dstLocalList3[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList3[NCHW_CONV_ADDR_LIST_SIZE];

    TransDataTo5HDParams transDataParams1;
    TransDataTo5HDParams transDataParams2;
    TransDataTo5HDParams transDataParams3;
};

template <typename T>
__aicore__ inline void InitConfusionTranspose2NZ012N(ConfusionTranspose2NZ012NParams<T> &params,
    ConfusionTranspose2NZ012NTiling &tiling)
{
    // number of lines required for initializing tmp2
    if (tiling.hnDiv < BLOCK_CUBE) {
        params.tmp2NeedRowCount = tiling.hnDiv;
    }

    params.tmp1RemainRowCount = 0;
    params.tmp2Count = 0;
    params.tmp2NeedRowCount = BLOCK_CUBE;
    params.transdataRepeat = 0;
    params.dstPrehnCount = 0;
    params.dstAllCount = 0;

    params.transDataParams1.repeatTimes = 1;
    params.transDataParams1.dstRepStride = 0;
    params.transDataParams1.srcRepStride = 0;

    params.transDataParams2.repeatTimes = 1;
    params.transDataParams2.dstRepStride = 0;
    params.transDataParams2.srcRepStride = 0;

    params.transDataParams3.repeatTimes = 1;
    params.transDataParams3.dstRepStride = 0;
    params.transDataParams3.srcRepStride = 0;
}

template <typename T>
__aicore__ inline void ConfusionTranspose2NZ012NTmp1RemainRowCountZero(const LocalTensor<T>& srcTensor,
    ConfusionTranspose2NZ012NTiling& tiling, ConfusionTranspose2NZ012NParams<T>& params, const LocalTensor<T>& tmp1)
{
    // src->tmp1 :transpose Column Fractal -- > Row Fractal
    if (params.tmp1RemainRowCount == 0) {
        if constexpr (sizeof(T) == sizeof(half)) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                params.dstLocalList1[n] = (uint64_t)tmp1[BLOCK_CUBE * n].GetPhyAddr();
            }
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                params.srcLocalList1[n] = (uint64_t)srcTensor[params.i * tiling.alignsBlockCube +
                    params.j * CUBE_MAX_SIZE + BLOCK_CUBE * n + params.k * tiling.srcBatchOffset]
                    .GetPhyAddr();
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
                        (uint64_t)srcTensor[params.i * tiling.alignsBlockCube + params.j * CUBE_MAX_SIZE +
                        m * tiling.blockSize + BLOCK_CUBE * n + params.k * tiling.srcBatchOffset]
                        .GetPhyAddr();
                }
                PipeBarrier<PIPE_V>();
                TransDataTo5HD<T>(params.dstLocalList1, params.srcLocalList1, params.transDataParams1);
            }
        }
        // Update the number of rows stored in tmp1
        params.tmp1RemainRowCount = BLOCK_CUBE;
    }
}

template <typename T>
__aicore__ inline void ConfusionTranspose2NZ012NTmp2RemainRowCountFirst(ConfusionTranspose2NZ012NTiling& tiling,
    ConfusionTranspose2NZ012NParams<T>& params, const LocalTensor<T>& tmp1, const LocalTensor<T>& tmp2)
{
    // hnDiv >= 16, update the number of rows required for tmp2.
    // this is mainly used for processing the block where each H/N boundary is located.
    if (((params.i * BLOCK_CUBE) <= tiling.hnDiv) && (((params.i + 1) * BLOCK_CUBE) > tiling.hnDiv)) {
        params.tmp2NeedRowCount = BLOCK_CUBE - tiling.gap;
    }
    // first tmp1->tmp2: Transfer the number of valid lines of dst to tmp2.
    if (params.tmp2NeedRowCount <= params.tmp1RemainRowCount) {
        if (params.tmp2NeedRowCount != 0) {
            DataCopyParams dataCopyParams1;
            dataCopyParams1.blockCount = 1;
            dataCopyParams1.blockLen = params.tmp2NeedRowCount * tiling.blockNum;
            dataCopyParams1.dstStride = 0;
            dataCopyParams1.srcStride = 0;
            PipeBarrier<PIPE_V>();
            DataCopy(tmp2[(params.tmp2Count * BLOCK_CUBE)], tmp1, dataCopyParams1);
            // update the number of rows stored in tmp1 and the number of rows required by tmp2.
            params.dstPrehnCount += params.tmp2NeedRowCount;
            params.dstAllCount += params.tmp2NeedRowCount;
            params.tmp2Count += params.tmp2NeedRowCount;
            params.tmp1RemainRowCount -= params.tmp2NeedRowCount;
            params.tmp2NeedRowCount = 0;
            if (params.dstPrehnCount == tiling.hnDiv) {
                params.dstPrehnCount = 0;
            }
        }
    }
}

template <typename T>
__aicore__ inline void ConfusionTranspose2NZ012NTmp2RemainRowCountZero(const LocalTensor<T>& dstTensor,
    ConfusionTranspose2NZ012NTiling& tiling, ConfusionTranspose2NZ012NParams<T>& params, const LocalTensor<T>& tmp2)
{
    // first tmp2 -> dst
    if (params.tmp2NeedRowCount == 0) {
        if constexpr (sizeof(T) == sizeof(half)) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                params.dstLocalList2[n] =
                    (uint64_t)
                        dstTensor[(params.transdataRepeat - params.j * tiling.prehBlockNum) * tiling.alignsBlockCube +
                    params.j * CUBE_MAX_SIZE + BLOCK_CUBE * n + params.k * tiling.dstBatchOffset]
                    .GetPhyAddr();
            }
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                params.srcLocalList2[n] = (uint64_t)tmp2[BLOCK_CUBE * n].GetPhyAddr();
            }
            PipeBarrier<PIPE_V>();
            TransDataTo5HD<T>(params.dstLocalList2, params.srcLocalList2, params.transDataParams2);
        } else if constexpr (sizeof(T) == sizeof(float)) {
            for (uint16_t m = 0; m < 2; m++) {
                for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                    params.dstLocalList2[n] =
                        (uint64_t)dstTensor[(params.transdataRepeat - params.j * tiling.prehBlockNum) *
                        tiling.alignsBlockCube +
                        params.j * CUBE_MAX_SIZE + m * CUBE_HALF_SIZE + tiling.blockSize * n +
                        params.k * tiling.dstBatchOffset]
                        .GetPhyAddr();
                }
                for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                    params.srcLocalList2[n] = (uint64_t)tmp2[m * tiling.blockSize + BLOCK_CUBE * n].GetPhyAddr();
                }
                PipeBarrier<PIPE_V>();
                TransDataTo5HD<T>(params.dstLocalList2, params.srcLocalList2, params.transDataParams2);
            }
        }
        params.transdataRepeat += 1;
        // Check whether it is the tail block of each horizontal block. If yes, the remaining invalid data of tmp1 does
        // not need to enter the internal loop. After the end of each row is transposed, the remaining part of tmp1 is
        // dirty data. In this case, tmp1RemainRowCount is set to 0.
        if ((params.transdataRepeat % tiling.shapeN) == 0 && (tiling.hnDiv < BLOCK_CUBE)) {
            params.tmp1RemainRowCount = 0;
        } else if ((params.dstAllCount % tiling.shapeH) == 0) {
            // After the valid data required by dst is transposed, the remaining part of tmp1 is dirty data. In this
            // case, tmp1RemainRowCount is set to 0.
            params.tmp1RemainRowCount = 0;
        }
        // Each time tmp2->dst is transposed, tmp2Count that points to the stored data in tmp2 is set to 0.
        params.tmp2Count = 0;
    }
}

template <typename T>
__aicore__ inline void ConfusionTranspose2NZ012NCalcTmp2NeedRowCount(ConfusionTranspose2NZ012NTiling& tiling,
    ConfusionTranspose2NZ012NParams<T>& params)
{
    // need tmp2NeedRowCount
    if ((tiling.hnDiv >= BLOCK_CUBE) && (params.dstPrehnCount == tiling.hnDiv)) {
        params.tmp2NeedRowCount = BLOCK_CUBE;
    } else if ((tiling.hnDiv >= BLOCK_CUBE) && (params.dstPrehnCount != tiling.hnDiv)) {
        params.tmp2NeedRowCount =
            (tiling.hnDiv - params.dstPrehnCount) >= BLOCK_CUBE ? BLOCK_CUBE : (tiling.hnDiv - params.dstPrehnCount);
    } else if (tiling.hnDiv < BLOCK_CUBE) {
        params.tmp2NeedRowCount = tiling.hnDiv;
    }
}

template <typename T>
__aicore__ inline void ConfusionTranspose2NZ012NTmp2NeedRowCount(ConfusionTranspose2NZ012NTiling& tiling,
    ConfusionTranspose2NZ012NParams<T>& params, const LocalTensor<T>& tmp1, const LocalTensor<T>& tmp2)
{
    // tmp1->tmp2
    if (params.tmp1RemainRowCount >= params.tmp2NeedRowCount) {
        if (params.tmp2NeedRowCount != 0) {
            DataCopyParams dataCopyParams2;
            dataCopyParams2.blockCount = 1;
            dataCopyParams2.blockLen = params.tmp2NeedRowCount * tiling.blockNum;
            dataCopyParams2.dstStride = 0;
            dataCopyParams2.srcStride = 0;
            PipeBarrier<PIPE_V>();
            DataCopy(tmp2, tmp1[((BLOCK_CUBE - params.tmp1RemainRowCount) * BLOCK_CUBE)], dataCopyParams2);
            params.dstPrehnCount += params.tmp2NeedRowCount;
            params.dstAllCount += params.tmp2NeedRowCount;
            params.tmp2Count = params.tmp2NeedRowCount;
            params.tmp1RemainRowCount -= params.tmp2NeedRowCount;
            params.tmp2NeedRowCount = 0;
            // Update dstPrehnCount. If dstPrehnCount is equal to hnDiv, the fractal processing of each horizontal row
            // is complete.
            if (params.dstPrehnCount == tiling.hnDiv) {
                params.dstPrehnCount = 0;
            }
        }
    } else {
        if (params.tmp2NeedRowCount != 0) {
            DataCopyParams dataCopyParams2;
            dataCopyParams2.blockCount = 1;
            dataCopyParams2.blockLen = params.tmp1RemainRowCount * tiling.blockNum;
            dataCopyParams2.dstStride = 0;
            dataCopyParams2.srcStride = 0;
            PipeBarrier<PIPE_V>();
            DataCopy(tmp2, tmp1[((BLOCK_CUBE - params.tmp1RemainRowCount) * BLOCK_CUBE)], dataCopyParams2);
            params.dstPrehnCount += params.tmp1RemainRowCount;
            params.dstAllCount += params.tmp1RemainRowCount;
            params.tmp2Count = params.tmp1RemainRowCount;
            params.tmp2NeedRowCount -= params.tmp1RemainRowCount;
            params.tmp1RemainRowCount = 0;
            if (params.dstPrehnCount == tiling.hnDiv) {
                params.dstPrehnCount = 0;
            }
        }
    }
}

template <typename T>
__aicore__ inline void ConfusionTranspose2NZ012NTmp2NeedRowCountZero(const LocalTensor<T>& dstTensor,
    ConfusionTranspose2NZ012NTiling& tiling, ConfusionTranspose2NZ012NParams<T>& params, const LocalTensor<T>& tmp2)
{
    // tmp2->dst
    if (params.tmp2NeedRowCount == 0) {
        if constexpr (sizeof(T) == sizeof(half)) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                params.dstLocalList3[n] =
                    (uint64_t)
                        dstTensor[(params.transdataRepeat - params.j * tiling.prehBlockNum) * tiling.alignsBlockCube +
                    params.j * CUBE_MAX_SIZE + BLOCK_CUBE * n + params.k * tiling.dstBatchOffset].GetPhyAddr();
            }
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                params.srcLocalList3[n] = (uint64_t)tmp2[BLOCK_CUBE * n].GetPhyAddr();
            }
            PipeBarrier<PIPE_V>();
            TransDataTo5HD<T>(params.dstLocalList3, params.srcLocalList3, params.transDataParams3);
        } else if constexpr (sizeof(T) == sizeof(float)) {
            for (uint16_t m = 0; m < 2; m++) {
                for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                    params.dstLocalList3[n] =
                        (uint64_t)dstTensor[(params.transdataRepeat - params.j * tiling.prehBlockNum) *
                        tiling.alignsBlockCube +
                        params.j * CUBE_MAX_SIZE + m * CUBE_HALF_SIZE + tiling.blockSize * n +
                        params.k * tiling.dstBatchOffset].GetPhyAddr();
                }
                for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                    params.srcLocalList3[n] = (uint64_t)tmp2[m * tiling.blockSize + BLOCK_CUBE * n].GetPhyAddr();
                }
                PipeBarrier<PIPE_V>();
                TransDataTo5HD<T>(params.dstLocalList3, params.srcLocalList3, params.transDataParams3);
            }
        }
        params.transdataRepeat += 1;
        if ((params.transdataRepeat % tiling.shapeN) == 0 && (tiling.hnDiv < BLOCK_CUBE)) {
            params.tmp1RemainRowCount = 0;
        } else if ((params.dstAllCount % tiling.shapeH) == 0) {
            params.tmp1RemainRowCount = 0;
        }

        params.tmp2Count = 0;

        if ((tiling.hnDiv >= BLOCK_CUBE) && (params.dstPrehnCount == tiling.hnDiv)) {
            params.tmp2NeedRowCount = BLOCK_CUBE;
        } else if ((tiling.hnDiv >= BLOCK_CUBE) && (params.dstPrehnCount != tiling.hnDiv)) {
            params.tmp2NeedRowCount = (tiling.hnDiv - params.dstPrehnCount) >= BLOCK_CUBE ?
                BLOCK_CUBE : (tiling.hnDiv - params.dstPrehnCount);
        } else if (tiling.hnDiv < BLOCK_CUBE) {
            params.tmp2NeedRowCount = tiling.hnDiv;
        }
    }
}

} // namespace AscendC
#endif // IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_BASE_2NZ012_H