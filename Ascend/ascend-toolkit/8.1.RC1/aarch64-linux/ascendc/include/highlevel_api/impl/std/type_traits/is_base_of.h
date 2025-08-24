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
 * \file is_base_of.h
 * \brief
 */
#ifndef IMPL_STD_ASCENDC_STD_IS_BASE_OF_IMPL__H
#define IMPL_STD_ASCENDC_STD_IS_BASE_OF_IMPL__H

#include "integral_constant.h"
#include "is_class.h"

namespace AscendC {
namespace Std {

template <typename Base, typename Derived>
struct IsBaseOfImpl {
private:
    template <typename B> 
    __aicore__ inline static true_type TestPtrConv(const volatile B *);

    template <typename B> 
    __aicore__ inline static false_type TestPtrConv(const volatile void *);

    template <typename B, typename D>
    __aicore__ inline static auto IsBaseOf(int32_t) -> decltype(TestPtrConv<B>(static_cast<D *>(nullptr)));

    template <typename B, typename D>
    __aicore__ inline static auto IsBaseOf(uint32_t) -> true_type; // private or ambiguous base

public:
    static constexpr bool value = decltype(IsBaseOf<Base, Derived>(0))::value;
};

template <typename Base, typename Derived>
struct is_base_of : bool_constant<is_class_v<Base> && is_class_v<Derived> && IsBaseOfImpl<Base, Derived>::value> {};

template <typename Base, typename Derived>
constexpr bool is_base_of_v = is_base_of<Base, Derived>::value;

}
}

#endif // IMPL_STD_ASCENDC_STD_IS_BASE_OF_IMPL__H
