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
 * \file is_tuple.h
 * \brief
 */
#ifndef IMPL_STD_ASCENDC_STD_IS_TUPLE_IMPL__H
#define IMPL_STD_ASCENDC_STD_IS_TUPLE_IMPL__H

#include "integral_constant.h"

namespace AscendC {
namespace Std {
 
template <typename T>
struct IsTupleImpl {
    private:
        template <typename Ts>
        __aicore__ inline static auto HasTupleSize(int32_t) -> bool_constant<(tuple_size<Ts>::value >= 0)>;

        template <typename Ts>
        __aicore__ inline static auto HasTupleSize(uint32_t) -> false_type;
    
    public:
        static constexpr bool value = decltype(HasTupleSize<T>((int32_t)0))::value;
};

template <typename T>
struct is_tuple : bool_constant<IsTupleImpl<T>::value> {};

template <typename T>
constexpr bool is_tuple_v = is_tuple<T>::value;
 
}
}

#endif // IMPL_STD_ASCENDC_STD_IS_TUPLE_IMPL__H