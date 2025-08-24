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
 * \file remove_cvref.h
 * \brief
 */
#ifndef IMPL_STD_ASCENDC_STD_REMOVE_CVREF_IMPL__H
#define IMPL_STD_ASCENDC_STD_REMOVE_CVREF_IMPL__H

#include "is_same.h"
#include "remove_cv.h"
#include "remove_reference.h"

namespace AscendC {
namespace Std {

template <typename Tp>
using remove_cvref_t = remove_cv_t<remove_reference_t<Tp>>;

template <typename Tp>
struct remove_cvref {
    using type = remove_cvref_t<Tp>;
};

template <typename Tp, typename Up>
struct is_same_uncvref : IsSame<remove_cvref_t<Tp>, remove_cvref_t<Up>> {};

}
}

#endif // IMPL_STD_ASCENDC_STD_REMOVE_CVREF_IMPL__H