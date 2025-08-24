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
 * \file is_convertible.h
 * \brief
 */
#ifndef IMPL_STD_ASCENDC_STD_IS_CONVERTIBLE_IMPL__H
#define IMPL_STD_ASCENDC_STD_IS_CONVERTIBLE_IMPL__H

#include "integral_constant.h"
#include "is_void.h"
#include "add_rvalue_reference.h"
#include "../utility/declval.h"

namespace AscendC {
namespace Std {

template <typename From, typename To>
struct IsConvertibleImpl {
private:
    template <typename T>
    __aicore__ inline static auto TestReturnable(int32_t) -> decltype(void(static_cast<T (*)()>(nullptr)), true_type{});

    template <typename T>
    __aicore__ inline static auto TestReturnable(uint32_t) -> false_type;

    template <typename F, typename T>
    __aicore__ inline static auto TestImplicitlyConvertible(int32_t) -> 
        decltype(void(declval<void (&)(T)>()(declval<F>())), true_type{});

    template <typename F, typename T>
    __aicore__ inline static auto TestImplicitlyConvertible(uint32_t) -> false_type;

public:
    static constexpr bool value = 
        decltype(TestReturnable<To>(0))::value && decltype(TestImplicitlyConvertible<From, To>(0))::value;
};

template <typename From, typename To>
struct is_convertible : bool_constant<(is_void_v<From> && is_void_v<To>) || IsConvertibleImpl<From, To>::value> {};

template <typename From, typename To>
constexpr bool is_convertible_v = is_convertible<From, To>::value;

template <typename Ty>
struct is_convertible<Ty&, volatile Ty&> : true_type {};

template <typename Ty>
struct is_convertible<volatile Ty&, volatile Ty&> : true_type {};

template <typename Ty>
struct is_convertible<Ty&, const volatile Ty&> : true_type {};

template <typename Ty>
struct is_convertible<volatile Ty&, const volatile Ty&> : true_type {};

template <typename Ty>
constexpr bool is_convertible_v<Ty&, volatile Ty&> = true;

template <typename Ty>
constexpr bool is_convertible_v<volatile Ty&, volatile Ty&> = true;

template <typename Ty>
constexpr bool is_convertible_v<Ty&, const volatile Ty&> = true;

template <typename Ty>
constexpr bool is_convertible_v<volatile Ty&, const volatile Ty&> = true;

}
}

#endif // IMPL_STD_ASCENDC_STD_IS_CONVERTIBLE_IMPL__H
