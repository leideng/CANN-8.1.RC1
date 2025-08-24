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
#ifndef IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_BASE_0213_H
#define IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_BASE_0213_H
#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_transpose_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_transpose_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_transpose_impl.h"
#endif
namespace AscendC {
const uint32_t CUBE_HALF_SIZE = CUBE_MAX_SIZE / 2;

template <typename T> struct ConfusionTranspose0213Params {
    __aicore__ ConfusionTranspose0213Params(){};

    int32_t i;
    int32_t j;
    int32_t k;
    int32_t m;

    TransposeType transposeType;

    TransDataTo5HDParams mainTransDataParams;
    TransDataTo5HDParams transDataParams1;
    TransDataTo5HDParams tailTransDataParams;
    TransDataTo5HDParams transDataParams2;

    // main : src->tmp1
    uint64_t mainDstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t mainSrcLocalList[NCHW_CONV_ADDR_LIST_SIZE];

    // main : tmp1->dst
    uint64_t dstLocalList1[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList1[NCHW_CONV_ADDR_LIST_SIZE];

    // tail : src->tmp1
    uint64_t tailDstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t tailSrcLocalList[NCHW_CONV_ADDR_LIST_SIZE];

    // tail : tmp1->dst
    uint64_t dstLocalList2[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList2[NCHW_CONV_ADDR_LIST_SIZE];
};

template <typename T>
__aicore__ inline void InitConfusionTranspose0213TransParams(ConfusionTranspose0213Tiling& tiling,
    ConfusionTranspose0213Params<T>& params, TransposeType transposeTypeIn)
{
    params.transposeType = transposeTypeIn;

    params.mainTransDataParams.repeatTimes = tiling.newPopSize / CUBE_MAX_SIZE;
    params.mainTransDataParams.dstRepStride = BLOCK_CUBE / tiling.blockSize;
    params.mainTransDataParams.srcRepStride = CUBE_MAX_SIZE / tiling.blockSize;
    if (params.mainTransDataParams.repeatTimes == 1) {
        params.mainTransDataParams.dstRepStride = 0;
        params.mainTransDataParams.srcRepStride = 0;
    }

    params.transDataParams1.repeatTimes = tiling.newPopSize / CUBE_MAX_SIZE;
    params.transDataParams1.dstRepStride = tiling.alignA3MulA1 * BLOCK_CUBE / tiling.blockSize;
    params.transDataParams1.srcRepStride = BLOCK_CUBE / tiling.blockSize;
    if (params.transDataParams1.repeatTimes == 1) {
        params.transDataParams1.dstRepStride = 0;
        params.transDataParams1.srcRepStride = 0;
    }

    params.tailTransDataParams.repeatTimes = tiling.tailSize / CUBE_MAX_SIZE;
    params.tailTransDataParams.dstRepStride = BLOCK_CUBE / tiling.blockSize;
    params.tailTransDataParams.srcRepStride = CUBE_MAX_SIZE / tiling.blockSize;
    if (params.tailTransDataParams.repeatTimes == 1) {
        params.tailTransDataParams.dstRepStride = 0;
        params.tailTransDataParams.srcRepStride = 0;
    }

    params.transDataParams2.repeatTimes = tiling.tailSize / CUBE_MAX_SIZE;
    params.transDataParams2.dstRepStride = tiling.alignA3MulA1 * BLOCK_CUBE / tiling.blockSize;
    params.transDataParams2.srcRepStride = BLOCK_CUBE / tiling.blockSize;
    if (params.transDataParams2.repeatTimes == 1) {
        params.transDataParams2.dstRepStride = 0;
        params.transDataParams2.srcRepStride = 0;
    }
}

template <typename T>
__aicore__ inline void ConfusionTranspose0213MainHalf(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    ConfusionTranspose0213Params<T>& params, ConfusionTranspose0213Tiling& tiling, const LocalTensor<T>& tmp1)
{
    // src--> tmp
    for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
        params.mainDstLocalList[n] = (uint64_t)tmp1[(tiling.newPopH * n)].GetPhyAddr();
    }
    for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
        params.mainSrcLocalList[n] = (uint64_t)srcTensor[(tiling.newPopSize * params.m + params.j * tiling.needSize +
            params.i * tiling.alignA2MulAlignA3 + BLOCK_CUBE * n + params.k * tiling.batchOffset)]
            .GetPhyAddr();
    }
    PipeBarrier<PIPE_V>();
    TransDataTo5HD<T>(params.mainDstLocalList, params.mainSrcLocalList, params.mainTransDataParams);
    // tmp --> dst
    if (params.transposeType == TransposeType::TRANSPOSE_NZ2ND_0213) {
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
            params.dstLocalList1[n] =
                (uint64_t)dstTensor[(tiling.newPopH * params.m * tiling.alignA3MulA1 + params.j * BLOCK_CUBE +
                params.i * tiling.alignA3 + tiling.alignA3MulA1 * n + params.k * tiling.batchOffset)]
                .GetPhyAddr();
        }
    } else if (params.transposeType == TransposeType::TRANSPOSE_NZ2NZ_0213) {
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
            params.dstLocalList1[n] =
                (uint64_t)
                    dstTensor[(tiling.newPopH * params.m * tiling.alignA3MulA1 + params.j * tiling.shapeA1BlockCube +
                params.i * BLOCK_CUBE + tiling.alignA3MulA1 * n + params.k * tiling.batchOffset)]
                .GetPhyAddr();
        }
    }
    for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
        params.srcLocalList1[n] = (uint64_t)tmp1[(tiling.newPopH * n)].GetPhyAddr();
    }
    PipeBarrier<PIPE_V>();
    TransDataTo5HD<T>(params.dstLocalList1, params.srcLocalList1, params.transDataParams1);
}

template <typename T>
__aicore__ inline void ConfusionTranspose0213MainFloat(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    ConfusionTranspose0213Params<T>& params, ConfusionTranspose0213Tiling& tiling, const LocalTensor<T>& tmp1)
{
    for (int16_t p = 0; p < 2; p++) {
        // src--> tmp
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n = n + 2) {
            params.mainDstLocalList[n] =
                (uint64_t)tmp1[(p * tiling.blockSize * tiling.newPopH + tiling.newPopH * (n / 2))].GetPhyAddr();
            params.mainDstLocalList[n + 1] =
                (uint64_t)tmp1[(p * tiling.blockSize * tiling.newPopH + tiling.newPopH * (n / 2)) + tiling.blockSize]
                .GetPhyAddr();
        }
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
            params.mainSrcLocalList[n] = (uint64_t)
                srcTensor[(p * tiling.blockSize + tiling.newPopSize * params.m + params.j * tiling.needSize +
                params.i * tiling.alignA2MulAlignA3 + BLOCK_CUBE * n + params.k * tiling.batchOffset)].GetPhyAddr();
        }
        PipeBarrier<PIPE_V>();
        TransDataTo5HD<T>(params.mainDstLocalList, params.mainSrcLocalList, params.mainTransDataParams);
    }
    for (int16_t p = 0; p < 2; p++) {
        // tmp --> dst
        if (params.transposeType == TransposeType::TRANSPOSE_NZ2ND_0213) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n = n + 2) {
                params.dstLocalList1[n] = (uint64_t)dstTensor[(p * tiling.blockSize * tiling.alignA3MulA1 +
                    tiling.newPopH * params.m * tiling.alignA3MulA1 + params.j * BLOCK_CUBE +
                    params.i * tiling.alignA3 + tiling.alignA3MulA1 * (n / 2) + params.k * tiling.batchOffset)]
                    .GetPhyAddr();
                params.dstLocalList1[n + 1] = (uint64_t)dstTensor[(p * tiling.blockSize * tiling.alignA3MulA1 +
                    tiling.newPopH * params.m * tiling.alignA3MulA1 + params.j * BLOCK_CUBE +
                    params.i * tiling.alignA3 + tiling.alignA3MulA1 * (n / 2) + tiling.blockSize +
                    params.k * tiling.batchOffset)].GetPhyAddr();
            }
        } else if (params.transposeType == TransposeType::TRANSPOSE_NZ2NZ_0213) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n = n + 2) {
                params.dstLocalList1[n] = (uint64_t)dstTensor[(p * tiling.blockSize * tiling.alignA3MulA1 +
                    tiling.newPopH * params.m * tiling.alignA3MulA1 + params.j * tiling.shapeA1BlockCube +
                    params.i * BLOCK_CUBE + tiling.alignA3MulA1 * (n / 2) + params.k * tiling.batchOffset)]
                    .GetPhyAddr();
                params.dstLocalList1[n + 1] = (uint64_t)dstTensor[(p * tiling.blockSize * tiling.alignA3MulA1 +
                    tiling.newPopH * params.m * tiling.alignA3MulA1 + params.j * tiling.shapeA1BlockCube +
                    params.i * BLOCK_CUBE + tiling.alignA3MulA1 * (n / 2) + tiling.blockSize +
                    params.k * tiling.batchOffset)].GetPhyAddr();
            }
        }
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
            params.srcLocalList1[n] = (uint64_t)tmp1[(p * tiling.blockSize + tiling.newPopH * n)].GetPhyAddr();
        }
        PipeBarrier<PIPE_V>();
        TransDataTo5HD<T>(params.dstLocalList1, params.srcLocalList1, params.transDataParams1);
    }
}

template <typename T>
__aicore__ inline void ConfusionTranspose0213TailHalf(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    ConfusionTranspose0213Params<T>& params, ConfusionTranspose0213Tiling& tiling, const LocalTensor<T>& tmp1)
{
    // src--> tmp
    for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
        params.tailDstLocalList[n] = (uint64_t)tmp1[tiling.newPopH * n].GetPhyAddr();
    }
    for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
        params.tailSrcLocalList[n] =
            (uint64_t)srcTensor[(tiling.newPopSize * tiling.mainBlocks + params.j * tiling.needSize +
            params.i * tiling.alignA2MulAlignA3 + BLOCK_CUBE * n + params.k * tiling.batchOffset)]
            .GetPhyAddr();
    }
    PipeBarrier<PIPE_V>();
    TransDataTo5HD<T>(params.tailDstLocalList, params.tailSrcLocalList, params.tailTransDataParams);
    // tmp --> dst
    if (params.transposeType == TransposeType::TRANSPOSE_NZ2ND_0213) {
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
            params.dstLocalList2[n] = (uint64_t)dstTensor[(tiling.mainOffset + params.j * BLOCK_CUBE +
                params.i * tiling.alignA3 + tiling.alignA3MulA1 * n + params.k * tiling.batchOffset)]
                .GetPhyAddr();
        }
    } else if (params.transposeType == TransposeType::TRANSPOSE_NZ2NZ_0213) {
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
            params.dstLocalList2[n] = (uint64_t)dstTensor[(tiling.mainOffset + params.j * tiling.shapeA1BlockCube +
                params.i * BLOCK_CUBE + tiling.alignA3MulA1 * n + params.k * tiling.batchOffset)]
                .GetPhyAddr();
        }
    }
    for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
        params.srcLocalList2[n] = (uint64_t)tmp1[(tiling.newPopH * n)].GetPhyAddr();
    }
    PipeBarrier<PIPE_V>();
    TransDataTo5HD<T>(params.dstLocalList2, params.srcLocalList2, params.transDataParams2);
}

template <typename T>
__aicore__ inline void ConfusionTranspose0213TailFloat(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    ConfusionTranspose0213Params<T>& params, ConfusionTranspose0213Tiling& tiling, const LocalTensor<T>& tmp1)
{
    for (int16_t p = 0; p < 2; p++) {
        // src--> tmp
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n = n + 2) {
            params.tailDstLocalList[n] =
                (uint64_t)tmp1[(p * tiling.blockSize * tiling.newPopH + tiling.newPopH * (n / 2))].GetPhyAddr();
            params.tailDstLocalList[n + 1] =
                (uint64_t)tmp1[(p * tiling.blockSize * tiling.newPopH + tiling.newPopH * (n / 2)) + tiling.blockSize]
                .GetPhyAddr();
        }
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
            params.tailSrcLocalList[n] = (uint64_t)srcTensor[(p * tiling.blockSize +
                tiling.newPopSize * tiling.mainBlocks + params.j * tiling.needSize +
                params.i * tiling.alignA2MulAlignA3 + BLOCK_CUBE * n + params.k * tiling.batchOffset)].GetPhyAddr();
        }
        PipeBarrier<PIPE_V>();
        TransDataTo5HD<T>(params.tailDstLocalList, params.tailSrcLocalList, params.tailTransDataParams);
    }
    for (int16_t p = 0; p < 2; p++) {
        // tmp --> dst
        if (params.transposeType == TransposeType::TRANSPOSE_NZ2ND_0213) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n = n + 2) {
                params.dstLocalList2[n] = (uint64_t)dstTensor[(p * tiling.blockSize * tiling.alignA3MulA1 +
                    tiling.mainOffset + params.j * BLOCK_CUBE + params.i * tiling.alignA3 +
                    tiling.alignA3MulA1 * (n / 2) + params.k * tiling.batchOffset)].GetPhyAddr();
                params.dstLocalList2[n + 1] = (uint64_t)dstTensor[(p * tiling.blockSize * tiling.alignA3MulA1 +
                    tiling.mainOffset + params.j * BLOCK_CUBE + params.i * tiling.alignA3 +
                    tiling.alignA3MulA1 * (n / 2) + tiling.blockSize + params.k * tiling.batchOffset)].GetPhyAddr();
            }
        } else if (params.transposeType == TransposeType::TRANSPOSE_NZ2NZ_0213) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n = n + 2) {
                params.dstLocalList2[n] = (uint64_t)dstTensor[(p * tiling.blockSize * tiling.alignA3MulA1 +
                    tiling.mainOffset + params.j * tiling.shapeA1BlockCube + params.i * BLOCK_CUBE +
                    tiling.alignA3MulA1 * (n / 2) + params.k * tiling.batchOffset)].GetPhyAddr();
                params.dstLocalList2[n + 1] = (uint64_t)dstTensor[(p * tiling.blockSize * tiling.alignA3MulA1 +
                    tiling.mainOffset + params.j * tiling.shapeA1BlockCube + params.i * BLOCK_CUBE +
                    tiling.alignA3MulA1 * (n / 2) + tiling.blockSize + params.k * tiling.batchOffset)].GetPhyAddr();
            }
        }
        for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
            params.srcLocalList2[n] = (uint64_t)tmp1[(p * tiling.blockSize + tiling.newPopH * n)].GetPhyAddr();
        }
        PipeBarrier<PIPE_V>();
        TransDataTo5HD<T>(params.dstLocalList2, params.srcLocalList2, params.transDataParams2);
    }
}

} // namespace AscendC
#endif // IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_BASE_0213_H