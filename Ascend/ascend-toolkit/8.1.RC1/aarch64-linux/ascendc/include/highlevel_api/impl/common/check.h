/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef IMPL_COMMON_CHECK_H
#define IMPL_COMMON_CHECK_H

namespace AscendC {

template <typename T>
__aicore__ inline void CheckTensorPosition(const LocalTensor<T> &checkTensor, __gm__ const char* tensorInfo,
    __gm__ const char* supportPosInfo)
{
#if ASCENDC_CPU_DEBUG
    ASCENDC_ASSERT(((TPosition)checkTensor.GetPosition() == TPosition::VECIN ||
        (TPosition)checkTensor.GetPosition() == TPosition::VECOUT ||
        (TPosition)checkTensor.GetPosition() == TPosition::VECCALC), {
        KERNEL_LOG(KERNEL_ERROR,
            "Failed to check tensor position of %s, current api support positions are %s, current position is %s.",
            tensorInfo, supportPosInfo,
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(checkTensor.GetPosition())).c_str());
    });
#endif
}

template <typename T>
__aicore__ inline void CheckCalCount(const uint32_t calCount, __gm__ const char* calCountInfo,
    const LocalTensor<T> &checkTensor, __gm__ const char* tensorInfo, __gm__ const char* apiInfo)
{
#if ASCENDC_CPU_DEBUG
    ASCENDC_ASSERT((calCount <= checkTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR,
            "The %s parameter cannot be %u, should not be larger than %s size %u in %s.", calCountInfo, calCount,
            tensorInfo, checkTensor.GetSize(), apiInfo);
    });
#endif
}

__aicore__ inline void CheckTmpBufferSize(const uint32_t checkBufferSize, const uint32_t compBufferSize,
    const uint32_t tmpBufferSize)
{
    ASCENDC_ASSERT((checkBufferSize > compBufferSize), {
        KERNEL_LOG( KERNEL_ERROR, "Insufficient temporary space, current operation is not enough, "
            "but only %u units are available, please check the host tiling.", tmpBufferSize);
    });
}

} // namespace AscendC
#endif // IMPL_COMMON_CHECK_H