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
 * \file confusion_transpose.h
 * \brief
 */
#ifndef LIB_TRANSPOSE_CONFUSION_TRANSPOSE_H
#define LIB_TRANSPOSE_CONFUSION_TRANSPOSE_H
#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../impl/transpose/confusion_transpose/confusion_transpose_common_impl.h"

#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 200
namespace AscendC {
#pragma begin_pipe(V)

/* **************************************************************************************************
 * ConfusionTranspose                                              *
 * ************************************************************************************************* */
/*
 * @ingroup ConfusionTranspose
 * @arrange and reshape the data from src to dst.
 * @param [out] dstTensor output LocalTensor
 * @param [in] srcTensor input LocalTensor
 * @param [in] sharedTmpBuffer tmp buffer LocalTensor
 * @param [in] transposeType
 * @param [in] tiling ConfusionTranspose tiling
 */
template <typename T>
__aicore__ inline void ConfusionTranspose(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, TransposeType transposeType, ConfusionTransposeTiling& tiling)
{
    ConfusionTransposeImpl<T>(dstTensor, srcTensor, sharedTmpBuffer, transposeType, tiling);
}

/* **************************************************************************************************
 * ConfusionTranspose                                              *
 * ************************************************************************************************* */
/*
 * @ingroup ConfusionTranspose
 * @arrange and reshape the data from src to dst.
 * @param [out] dstTensor output LocalTensor
 * @param [in] srcTensor input LocalTensor
 * @param [in] transposeType
 * @param [in] tiling ConfusionTranspose tiling
 */
template <typename T>
__aicore__ inline void ConfusionTranspose(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    TransposeType transposeType, ConfusionTransposeTiling& tiling)
{
    LocalTensor<uint8_t> tmpBuffer;
    bool res = PopStackBuffer<uint8_t, TPosition::LCM>(tmpBuffer);
    ASCENDC_ASSERT(res, { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    ConfusionTransposeImpl<T>(dstTensor, srcTensor, tmpBuffer, transposeType, tiling);
}
#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_TRANSPOSE_CONFUSION_TRANSPOSE_H