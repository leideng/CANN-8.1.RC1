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
 * \file reduce_xor_sum_v200_impl.h
 * \brief
 */
#ifndef IMPL_REDUCE_REDUCE_XOR_SUM_REDUCE_XOR_SUM_V200_IMPL_H
#define IMPL_REDUCE_REDUCE_XOR_SUM_REDUCE_XOR_SUM_V200_IMPL_H
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
__aicore__ inline void CastInt162Float(const LocalTensor<float>& dst, const LocalTensor<int16_t>& src)
{
    Cast<half, int16_t, false>(src.ReinterpretCast<half>(), src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                               {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    Cast<float, half, false>(dst, src.ReinterpretCast<half>(), RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                             {1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE});
}

__aicore__ inline void CastFloat2Int16(const LocalTensor<int16_t>& dst, const LocalTensor<float>& src)
{
    Cast<half, float, false>(dst.ReinterpretCast<half>(), src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                             {1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    Cast<int16_t, half, false>(dst, dst.ReinterpretCast<half>(), RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1,
                               {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
}
} //  namespace AscendC
#endif // IMPL_REDUCE_REDUCE_XOR_SUM_REDUCE_XOR_SUM_V200_IMPL_H
