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
 * \file type_traits.h
 * \brief
 */
#ifndef LIB_STD_ASCENDC_STD_TYPE_TRAITS__H
#define LIB_STD_ASCENDC_STD_TYPE_TRAITS__H

#include "../../impl/std/type_traits/add_const.h"
#include "../../impl/std/type_traits/add_cv.h"
#include "../../impl/std/type_traits/add_lvalue_reference.h"
#include "../../impl/std/type_traits/add_pointer.h"
#include "../../impl/std/type_traits/add_rvalue_reference.h"
#include "../../impl/std/type_traits/add_volatile.h"
#include "../../impl/std/type_traits/conditional.h"
#include "../../impl/std/type_traits/decay.h"
#include "../../impl/std/type_traits/enable_if.h"
#include "../../impl/std/type_traits/integral_constant.h"
#include "../../impl/std/type_traits/is_array.h"
#include "../../impl/std/type_traits/is_base_of.h"
#include "../../impl/std/type_traits/is_class.h"
#include "../../impl/std/type_traits/is_constant.h"
#include "../../impl/std/type_traits/is_const.h"
#include "../../impl/std/type_traits/is_convertible.h"
#include "../../impl/std/type_traits/is_floating_point.h"
#include "../../impl/std/type_traits/is_function.h"
#include "../../impl/std/type_traits/is_integral.h"
#include "../../impl/std/type_traits/is_pointer.h"
#include "../../impl/std/type_traits/is_referenceable.h"
#include "../../impl/std/type_traits/is_reference.h"
#include "../../impl/std/type_traits/is_same.h"
#include "../../impl/std/type_traits/is_tuple.h"
#include "../../impl/std/type_traits/is_union.h"
#include "../../impl/std/type_traits/is_void.h"
#include "../../impl/std/type_traits/remove_const.h"
#include "../../impl/std/type_traits/remove_cv.h"
#include "../../impl/std/type_traits/remove_cvref.h"
#include "../../impl/std/type_traits/remove_extent.h"
#include "../../impl/std/type_traits/remove_pointer.h"
#include "../../impl/std/type_traits/remove_reference.h"
#include "../../impl/std/type_traits/remove_volatile.h"

namespace AscendC {
namespace Std {

// enable_if
template <bool, typename Tp>
struct enable_if;

// conditional
template <bool Bp, typename If, typename Then>
struct conditional;

// is_convertible
template <typename From, typename To>
struct is_convertible;

// is_base_of
template <typename Base, typename Derived>
struct is_base_of;

// is_same
template <typename Tp, typename Up>
struct is_same;

}
}

#endif // LIB_STD_ASCENDC_STD_TYPE_TRAITS__H
