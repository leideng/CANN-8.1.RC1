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
 * \file is_class.h
 * \brief
 */
#ifndef IMPL_STD_ASCENDC_STD_IS_CLASS_IMPL__H
#define IMPL_STD_ASCENDC_STD_IS_CLASS_IMPL__H

#include "integral_constant.h"
#include "is_union.h"

namespace AscendC {
namespace Std {

namespace IsClassImpl {

template <typename Tp>
__aicore__ inline bool_constant<!is_union_v<Tp>> Test(int32_t Tp::*);

template <typename Tp>
__aicore__ inline false_type Test(uint32_t);

} // namespace is_class_impl

template <typename Tp>
struct is_class : decltype(IsClassImpl::Test<Tp>(nullptr)) {};

template <typename Tp>
constexpr bool is_class_v = is_class<Tp>::value;

}
}

#endif // IMPL_STD_ASCENDC_STD_IS_CLASS_IMPL__H
