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
 * \file cosh_v200_impl.h
 * \brief
 */
#ifndef IMPL_MATH_COSH_COSH_V200_IMPL_H
#define IMPL_MATH_COSH_COSH_V200_IMPL_H
#include "kernel_tensor.h"

namespace AscendC {
__aicore__ inline void CoshCast(const LocalTensor<half>& dst, const LocalTensor<float>& src)
{
    Cast<half, float, false>(dst, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}
} // namespace AscendC

#endif // IMPL_MATH_COSH_COSH_V200_IMPL_H
