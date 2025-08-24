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
 * \file is_reference.h
 * \brief
 */
#ifndef IMPL_STD_ASCENDC_STD_IS_REFERENCE_IMPL__H
#define IMPL_STD_ASCENDC_STD_IS_REFERENCE_IMPL__H

#include "integral_constant.h"

namespace AscendC {
namespace Std {

template <typename Tp>
struct is_lvalue_reference : public false_type {};

template <typename Tp>
struct is_lvalue_reference<Tp&> : public true_type {};

template <typename Tp>
struct is_rvalue_reference : public false_type {};

template <typename Tp>
struct is_rvalue_reference<Tp&&> : public true_type {};

template <typename Tp>
struct is_reference : public false_type {};

template <typename Tp>
struct is_reference<Tp&> : public true_type {};

template <typename Tp>
struct is_reference<Tp&&> : public true_type {};

template <typename Tp>
constexpr bool is_lvalue_reference_v = is_lvalue_reference<Tp>::value;

template <typename Tp>
constexpr bool is_rvalue_reference_v = is_rvalue_reference<Tp>::value;

template <typename Tp>
constexpr bool is_reference_v = is_reference<Tp>::value;

}
}

#endif // IMPL_STD_ASCENDC_STD_IS_REFERENCE_IMPL__H
