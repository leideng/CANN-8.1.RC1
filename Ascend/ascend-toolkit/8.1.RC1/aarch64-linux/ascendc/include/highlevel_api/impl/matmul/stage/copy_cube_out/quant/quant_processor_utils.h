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
 * \file quant_processor_utils.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_UTILS_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_UTILS_H

namespace AscendC {
namespace Impl {
namespace Detail {
    
template <typename L0cT, typename DstT>
__aicore__ inline constexpr static bool IsQuantSenario()
{
    if constexpr (IsSameType<L0cT, int32_t>::value && IsSameType<DstT, half>::value) {
        return true;
    } else if constexpr (IsSameType<L0cT, int32_t>::value &&
        (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value)) {
        return true;
    } else if constexpr (IsSameType<L0cT, float>::value &&
        (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value)) {
        return true;
    }
    return false;
}
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_UTILS_H