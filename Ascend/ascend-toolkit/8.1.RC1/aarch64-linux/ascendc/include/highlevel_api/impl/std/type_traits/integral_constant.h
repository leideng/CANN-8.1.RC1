/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. 2025
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file integral_constant.h
 * \brief
 */
#ifndef IMPL_STD_ASCENDC_STD_INTEGRAL_CONSTANT_IMPL__H
#define IMPL_STD_ASCENDC_STD_INTEGRAL_CONSTANT_IMPL__H

namespace AscendC {
namespace Std {

template <typename Tp, Tp v>
struct integral_constant
{
    static constexpr const Tp value = v;

    using value_type = Tp;
    using type = integral_constant;

    __aicore__ inline constexpr operator value_type() const noexcept {
        return value;
    }

    __aicore__ inline constexpr value_type operator()() const noexcept {
        return value;
    }
};

template <typename Tp, Tp v>
constexpr const Tp integral_constant<Tp, v>::value;

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <bool b>
using bool_constant = integral_constant<bool, b>;

}
}

#endif // IMPL_STD_ASCENDC_STD_INTEGRAL_CONSTANT_IMPL__H
