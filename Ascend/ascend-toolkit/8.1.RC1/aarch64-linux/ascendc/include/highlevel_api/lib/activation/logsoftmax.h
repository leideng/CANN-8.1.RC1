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
 * \file logsoftmax.h
 * \brief LogSoftMax api of AscendC
 */
#ifndef LIB_SOFTMAX_LOGSOFTMAX_H
#define LIB_SOFTMAX_LOGSOFTMAX_H

#include "kernel_tensor.h"
#if __CCE_AICORE__ >= 200
#include "../../impl/activation/softmax/softmax_common.h"
#include "../../impl/activation/softmax/logsoftmax_base_impl.h"
#pragma begin_pipe(V)

namespace AscendC {
/*!
 * \ingroup LogSoftMax
 * \brief compute process: max = rowmax(x), sum = rowsum(exp(x-max)), y = log(exp(x-max)/sum)
 * \note support data type: half and float
 * \param [out] dst: output y
 * \param [out] sumTensor: output sum
 * \param [out] maxTensor: output max
 * \param [in] src: input x
 * \param [in] sharedTmpBuffer: input local temporary Tensor,you can get the range by tilingfunc of
 *                               GetSoftMaxMinTmpSize/GetSoftMaxMaxTmpSize
 * \param [in] softmaxShapeInfo: input src shape
 * \param [in] tiling: input softmaxtiling
 * \param [in] isReuseSource: reserved param
 * \param [in] isDataFormatNZ: if the data format of input src is NZ
 */
template <typename T, bool isReuseSource = false, bool isDataFormatNZ = false>
__aicore__ inline void LogSoftMax(LocalTensor<T>& dst, const LocalTensor<T>& sumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& src, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LogSoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }

    TRACE_START(TraceId::LogSoftMax);
    LogSoftMaxImpl<T, isReuseSource, isDataFormatNZ>(dst, sumTensor, maxTensor, src, sharedTmpBuffer,
        tiling, softmaxShapeInfo);
    TRACE_STOP(TraceId::LogSoftMax);
}
} // namespace AscendC

#pragma end_pipe
#endif
#endif // LIB_SOFTMAX_LOGSOFTMAX_H