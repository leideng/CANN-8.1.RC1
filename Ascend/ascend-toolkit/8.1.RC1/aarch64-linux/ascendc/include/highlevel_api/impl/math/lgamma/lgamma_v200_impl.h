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
 * \file lgamma_v200_impl.h
 * \brief
 */
#ifndef IMPL_MATH_LGAMMA_LGAMMA_V200_IMPL_H
#define IMPL_MATH_LGAMMA_LGAMMA_V200_IMPL_H
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"

#if __CCE_AICORE__ == 200
namespace AscendC {
__aicore__ inline void LGammaFloor(const LocalTensor<float> &dst, const LocalTensor<float> &src)
{
    Cast<int32_t, float, false>(dst.ReinterpretCast<int32_t>(), src, RoundMode::CAST_FLOOR, MASK_PLACEHOLDER, 1,
        {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    Cast<float, int32_t, false>(dst, dst.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
}
}  // namespace AscendC
#endif

#endif  // IMPL_MATH_LGAMMA_LGAMMA_V200_IMPL_H
