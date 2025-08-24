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
 * \file is_function.h
 * \brief
 */
#ifndef IMPL_STD_ASCENDC_STD_IS_FUNCTION_IMPL__H
#define IMPL_STD_ASCENDC_STD_IS_FUNCTION_IMPL__H

#include "integral_constant.h"
#include "is_const.h"
#include "is_reference.h"

namespace AscendC {
namespace Std {

template <typename T>
struct is_function : public bool_constant<!(is_reference_v<T> || is_const_v<const T>)> {};

template <typename Tp>
constexpr bool is_function_v = is_function<Tp>::value;

}
}

#endif // IMPL_STD_ASCENDC_STD_IS_FUNCTION_IMPL__H
