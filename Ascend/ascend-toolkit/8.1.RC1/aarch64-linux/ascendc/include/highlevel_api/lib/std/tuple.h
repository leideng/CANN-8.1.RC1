/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd. 2024
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file tuple.h
 * \brief
 */
#ifndef LIB_STD_ASCENDC_STD_TUPLE__H
#define LIB_STD_ASCENDC_STD_TUPLE__H

#include "../../impl/std/tuple/tuple_impl.h"

namespace AscendC {
namespace Std {

// tuple
template <typename ...Tps>
class tuple;

// tuple_size
template <typename ...Tps>
struct tuple_size;

// tuple_element
template <size_t N, typename ...Tps>
struct tuple_element;

// make_tuple
template <typename ...Tps>
__aicore__ inline constexpr tuple<unwrap_decay_t<Tps>...> make_tuple(Tps&& ...args);

// tie
template <typename ...Tps>
__aicore__ inline constexpr tuple<Tps& ...> tie(Tps& ...args) noexcept;

// get
template <size_t N, typename ...Tps>
__aicore__ inline typename tuple_element<N, tuple<Tps...> >::type& get(tuple<Tps...>& t) noexcept;

template <size_t N, typename ...Tps>
__aicore__ inline const typename tuple_element<N, tuple<Tps...> >::type& get(const tuple<Tps...>& t) noexcept;

template <size_t N, typename ...Tps>
__aicore__ inline typename tuple_element<N, tuple<Tps...> >::type&& get(tuple<Tps...>&& t) noexcept;

template <size_t N, typename ...Tps>
__aicore__ inline const typename tuple_element<N, tuple<Tps...> >::type&& get(const tuple<Tps...>&& t) noexcept;

}
}

#endif // LIB_STD_ASCENDC_STD_TUPLE__H
