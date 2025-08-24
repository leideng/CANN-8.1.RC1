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
 * \file softmax_flashv2_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_V200_SOFTMAX_FLASHV2_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_V200_SOFTMAX_FLASHV2_IMPL_H

#include "softmax_impl.h"
#include "../common/softmax_flashv2_impl/softmax_flashv2_basic_block_impl.h"
#include "../common/softmax_flashv2_impl/softmax_flashv2_update_impl.h"
#include "../common/softmax_flashv2_impl/softmax_flashv2_no_update_impl.h"
#include "../common/softmax_flashv2_impl/softmax_flashv2_nz_impl.h"
#include "../common/softmax_flashv2_impl/softmax_flashv2_common_impl.h"
namespace AscendC {

template <typename T1, typename T2, bool isUpdate = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashV2M1PostProcess(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& expSumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
    const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "softmaxflashv2 is not supported on current device!"); });
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_V200_SOFTMAX_FLASHV2_IMPL_H