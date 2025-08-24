/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file math_common_util.h
 * \brief defined common used math related function.
 */
#ifndef IMPL_MATH_MATH_COMMON_UTIL_H
#define IMPL_MATH_MATH_COMMON_UTIL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"

namespace AscendC {
namespace Internal {

template <typename T>
__aicore__ inline void CommonCheckInputsValidness(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                        const uint32_t calCount) {
#if ASCENDC_CPU_DEBUG
    ASCENDC_ASSERT(((TPosition)dstTensor.GetPosition() == TPosition::VECIN ||
                    (TPosition)dstTensor.GetPosition() == TPosition::VECOUT ||
                    (TPosition)dstTensor.GetPosition() == TPosition::VECCALC),
                  {
                    KERNEL_LOG(KERNEL_ERROR,
                                "dst position not support, just support position is VECIN, VECOUT, VECCALC.");
                  });

    ASCENDC_ASSERT((calCount <= srcTensor.GetSize()), {
      KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should not larger than srcTensor size %u", calCount,
                srcTensor.GetSize());
    });
#endif
}
} // namespace Internal
} // namespace AscendC

#endif // IMPL_MATH_MATH_COMMON_UTIL_H