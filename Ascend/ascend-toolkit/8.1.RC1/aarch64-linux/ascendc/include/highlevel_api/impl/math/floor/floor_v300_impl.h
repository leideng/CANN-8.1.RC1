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
 * \file floor_v300_impl.h
 * \brief
 */
#ifndef IMPL_MATH_FLOOR_FLOOR_V300_IMPL_H
#define IMPL_MATH_FLOOR_FLOOR_V300_IMPL_H
#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/check.h"

namespace AscendC {
__aicore__ inline void FloorProcess(const LocalTensor<half> &dstTensor, const LocalTensor<half> &srcTensor,
    const LocalTensor<uint8_t> &tmpTensor, const uint32_t calCount)
{
    const LocalTensor<float> floatTmpTensor = tmpTensor.ReinterpretCast<float>();

    // In the case of the half data type, there is no direct instruction for the round operation. Therefore, multiple
    // conversions are required.
    Cast<float, half>(floatTmpTensor, srcTensor, RoundMode::CAST_NONE, calCount);

    Cast<float, float>(floatTmpTensor, floatTmpTensor, RoundMode::CAST_FLOOR, calCount);

    Cast<half, float>(dstTensor, floatTmpTensor, RoundMode::CAST_NONE, calCount);
}
__aicore__ inline void FloorImpl(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Floor");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Floor");

    // Calculate the amount of data that can be stored in the temporary space and split the data into the entire block
    // and tail block.
    uint32_t tmpBufferSize = sharedTmpBuffer.GetSize();
    uint32_t splitCount = tmpBufferSize / sizeof(float) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    CheckTmpBufferSize(splitCount, 0, tmpBufferSize);

    uint32_t loopCount = calCount / splitCount;
    uint32_t calcTail = calCount % splitCount;

    for (uint32_t i = 0; i < loopCount; ++i) {
        FloorProcess(dstTensor[i * splitCount], srcTensor[i * splitCount], sharedTmpBuffer, splitCount);
    }
    if (calcTail > 0) {
        FloorProcess(dstTensor[loopCount * splitCount], srcTensor[loopCount * splitCount], sharedTmpBuffer, calcTail);
    }
}
__aicore__ inline void FloorImpl(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor,
    const uint32_t calCount)
{
    // alloc tmp buffer using stack
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    FloorImpl(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}
__aicore__ inline void FloorImpl(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Floor");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Floor");

    (void)sharedTmpBuffer;
    Cast<float, float>(dstTensor, srcTensor, RoundMode::CAST_FLOOR, calCount);
}
__aicore__ inline void FloorImpl(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const uint32_t calCount)
{
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Floor");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Floor");

    Cast<float, float>(dstTensor, srcTensor, RoundMode::CAST_FLOOR, calCount);
}
}  // namespace AscendC
#endif  // IMPL_MATH_FLOOR_FLOOR_V300_IMPL_H