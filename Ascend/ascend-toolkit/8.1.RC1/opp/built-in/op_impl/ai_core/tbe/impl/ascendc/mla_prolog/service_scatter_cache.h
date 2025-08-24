/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

/*!
 * \file service_scatter_cache.h
 * \brief
 */

#ifndef SERVICE_SCATTER_CACHE_H
#define SERVICE_SCATTER_CACHE_H

#include "mla_prolog_comm.h"

namespace MlaProlog {

/**
 * @brief 非PA场景，将inputLocal中的数据scatter到cacheGm，只支持ND cache
 * @param inputLocal 输入tensor，[row, col]，一行对应一个token，只支持单行数据处理，row为1
 * @param cacheGm 输出tensor，[B, S2, col]，[S2, col]对应一个batch的cache
 * @param cacheLength S2
 * @param batchIndex 待处理token在cache中的dim0下标，取值[0, B)
 * @param tokenIndexPerBatch 待处理token在cache中的dim1下标，取值[0, S2)
 * @param row 待处理的行数
 * @param col 待处理的列数，需满足32 bytes对齐
 */

template <typename T>
__aicore__ inline void ScatterCache(const LocalTensor<T>& inputLocal, GlobalTensor<T>& cacheGm, int64_t cacheLength,
                                    int64_t batchIndex, int64_t tokenIndexPerBatch, int64_t row, int64_t col) {
    DataCopy(cacheGm[(batchIndex * cacheLength + tokenIndexPerBatch) * col], inputLocal, col);
}

/**
 * @brief PA场景，将inputLocal中的数据scatter到cacheGm，支持ND和Nz cache
 * @param inputLocal 输入tensor，[row, col]，一行对应一个token，只支持单行数据处理，row为1
 * @param cacheGm 输出tensor
 *     ND [blockNum, blockSize, col]
 *     Nz [blockNum, ceil(col/col0), blockSize, col0]
 * @param blockSize KV blocks的大小
 * @param paTokenIndex 待处理token在cache中的全局index，取值[0, blockNum*blockSize)
 * @param row 待处理的行数
 * @param col 待处理的列数，需满足32 bytes对齐
 */

template <typename T, bool isNz>
__aicore__ inline void ScatterCache(const LocalTensor<T>& inputLocal, GlobalTensor<T>& cacheGm,
                                    int64_t blockSize, int64_t paTokenIndex, int64_t row, int64_t col) {
    if (paTokenIndex < 0) {
        return;
    }
    if constexpr (!isNz) {
        DataCopy(cacheGm[paTokenIndex * col], inputLocal, col);
    } else {
        constexpr uint8_t ALIGN_BLOCK_SIZE = 32;
        constexpr uint8_t col0 = ALIGN_BLOCK_SIZE / sizeof(T);
        int64_t cacheOffset = paTokenIndex / blockSize * blockSize * col + paTokenIndex % blockSize * col0;
        DataCopyParams copyParams {static_cast<uint16_t>(col / col0), 1, 0, static_cast<uint16_t>(blockSize - 1)};
        DataCopy(cacheGm[cacheOffset], inputLocal, copyParams);
    }
}

}

#endif