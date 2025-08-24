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
 * \file pad.h
 * \brief
 */
#ifndef LIB_PAD_PAD_H
#define LIB_PAD_PAD_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../impl/pad/pad/pad_common_impl.h"
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
namespace AscendC {
/* **************************************************************************************************
 * Pad                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Pad
 * @brief pad from src to dst, applicable to vector data
 * @param [out] dstTensor output LocalTensor
 * @param [in] srcTensor input LocalTensor
 * @param [in] sharedTmpBuffer tmp buffer LocalTensor
 * @param [in] PadParams.leftPad number of left pad
 * @param [in] PadParams.rightPad number of right pad
 * @param [in] PadParams.padValue value of pad
 */
#pragma begin_pipe(V)
template <typename T>
__aicore__ inline void Pad(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, PadParams &padParams,
    const LocalTensor<uint8_t> &sharedTmpBuffer, PadTiling &tiling)
{
    TRACE_START(TraceId::Pad);
    PadImpl<T>(dstTensor, srcTensor, padParams, sharedTmpBuffer, tiling);
    TRACE_STOP(TraceId::Pad);
}

/* **************************************************************************************************
 * Pad                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Pad
 * @brief pad from src to dst, applicable to vector data
 * @param [out] dstTensor output LocalTensor
 * @param [in] srcTensor input LocalTensor
 * @param [in] PadParams.leftPad number of left pad
 * @param [in] PadParams.rightPad number of right pad
 * @param [in] PadParams.padValue value of pad
 */
template <typename T>
__aicore__ inline void Pad(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, PadParams &padParams,
    PadTiling &tiling)
{
    LocalTensor<uint8_t> tmpBuffer;
    bool res = PopStackBuffer<uint8_t, TPosition::LCM>(tmpBuffer);
    ASCENDC_ASSERT(res, { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    PadImpl<T>(dstTensor, srcTensor, padParams, tmpBuffer, tiling);
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
__aicore__ inline void UnPad(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, UnPadParams &unPadParams,
     LocalTensor<uint8_t> &sharedTmpBuffer, UnPadTiling &tiling)
{
    UnPadImpl<T>(dstTensor, srcTensor, unPadParams, sharedTmpBuffer, tiling);
}

/* **************************************************************************************************
 * UnPad                                             *
 * ************************************************************************************************* */
/*
 * @ingroup UnPad
 * @brief unpad from src to dst, applicable to vector data
 * @param [out] dstTensor output LocalTensor
 * @param [in] srcTensor input LocalTensor
 * @param [in] unPadParams.leftPad number of left unpad
 * @param [in] unPadParams.rightPad number of right unpad
 */
template <typename T>
__aicore__ inline void UnPad(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, UnPadParams &unPadParams,
    UnPadTiling &tiling)
{
    LocalTensor<uint8_t> tmpBuffer;
    bool res = PopStackBuffer<uint8_t, TPosition::LCM>(tmpBuffer);
    ASCENDC_ASSERT(res, { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    UnPadImpl<T>(dstTensor, srcTensor, unPadParams, tmpBuffer, tiling);
}
#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_PAD_PAD_H
