/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. 2025
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max.h
 * \brief
 */
#ifndef IMPL_STD_ALGORITHM_MAX_H
#define IMPL_STD_ALGORITHM_MAX_H
#include "../type_traits/is_same.h"

namespace AscendC {
namespace Std {
template <typename T, typename U>
__aicore__ inline T max(const T src0, const U src1)
{
    static_assert(Std::is_same<T, U>::value, "Only support compare with same type!");
    return (src0 > src1) ? src0 : src1;
}

}
}
#endif  // IMPL_STD_ALGORITHM_MAX_H