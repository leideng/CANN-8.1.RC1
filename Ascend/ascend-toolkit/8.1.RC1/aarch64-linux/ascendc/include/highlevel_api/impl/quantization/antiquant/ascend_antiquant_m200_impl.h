/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ascend_antiquant_m200_impl.h
 * \brief
 */
#ifndef IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_M200_IMPL_H
#define IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_M200_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "ascend_antiquant_common.h"

namespace AscendC {

template <typename SrcType, typename OutType>
__aicore__ inline void CheckApiDtypeValid()
{
    constexpr bool inputValid = (IsSameType<SrcType, int8_t>::value);
    constexpr bool outputValid = (IsSameType<OutType, half>::value);
    ASCENDC_ASSERT((inputValid && outputValid), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in AscendAntiQuant, "
        "current api support dtype combination is src: int8_t, dst: half.");});
}

template <typename DstType>
__aicore__ inline bool AntiQuantCheckPerformanceMode(const LocalTensor<DstType> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K)
{
    return true;    // DstType can only be FP16, no need for cast
}

__aicore__ inline void AntiQuantFp16BrcbWithTransdata(const LocalTensor<half> &dst, const LocalTensor<half> &src,
    const uint32_t scaleN)
{
    __ubuf__ half* dstAddr = (__ubuf__  half*)dst.GetPhyAddr();
    uint64_t srcAddr = (uint64_t)(src.GetPhyAddr());
    uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE];
    for (uint32_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; ++i) {
        dstList[i] = (uint64_t)(dstAddr + B16_DATA_NUM_PER_BLOCK * i);
        srcList[i] = srcAddr;
    }
    const uint32_t repTimes = scaleN / B16_DATA_NUM_PER_BLOCK;
    uint16_t dstRepStride = DEFAULT_REPEAT_STRIDE * ANTIQUANT_TWO;
    uint16_t srcRepStride = 1;
    if (repTimes == 1) {
        dstRepStride = 0;
        srcRepStride = 0;
    }
    TransDataTo5HDParams params(false, false, repTimes, dstRepStride, srcRepStride);
    // broadcast element to block using transdataTo5HD
    TransDataTo5HD<half>(dstList, srcList, params);
    PipeBarrier<PIPE_V>();
}

template <bool withOffset = true>
__aicore__ inline void AntiQuantFp16Brcb(const LocalTensor<half> &scale, const LocalTensor<half> &offset,
    AntiquantParams<half> &params, uint32_t scaleN)
{
    AntiQuantFp16BrcbWithTransdata(params.tempTensorScale, scale, scaleN);
    if constexpr (withOffset) {
        AntiQuantFp16BrcbWithTransdata(params.tempTensorOffset, offset, scaleN);
    }
}

template <typename SrcType, typename DstType>
__aicore__ inline void AscendAntiQuantBF16Transpose(const LocalTensor<DstType> &dst, const LocalTensor<SrcType> &src,
    const LocalTensor<DstType> &offset, const LocalTensor<DstType> &scale, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const uint32_t K, const AntiQuantShapeInfo& shapeInfo = {})
{
    return;    // BF16 is not supported in current platform
}

template <typename SrcType, typename DstType>
__aicore__ inline void AscendAntiQuantBF16Transpose(const LocalTensor<DstType> &dst, const LocalTensor<SrcType> &src,
    const LocalTensor<DstType> &scale, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K,
    const AntiQuantShapeInfo& shapeInfo = {})
{
    return;    // BF16 is not supported in current platform
}
} // namespace AscendC
#endif // IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_M200_IMPL_H
