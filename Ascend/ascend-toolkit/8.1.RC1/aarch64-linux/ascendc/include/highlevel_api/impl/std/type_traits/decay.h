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
 * \file decay.h
 * \brief
 */
#ifndef IMPL_STD_ASCENDC_STD_DECAY_IMPL__H
#define IMPL_STD_ASCENDC_STD_DECAY_IMPL__H

#include "add_pointer.h"
#include "conditional.h"
#include "is_array.h"
#include "is_function.h"
#include "is_referenceable.h"
#include "remove_cv.h"
#include "remove_extent.h"
#include "remove_reference.h"

namespace AscendC {
namespace Std {

template <typename Up, bool>
struct DecayImpl {
    using type = remove_cv_t<Up>;
};

template <typename Up>
struct DecayImpl<Up, true>
{
public:
    using type = conditional_t<is_array<Up>::value, remove_extent_t<Up>*,
    conditional_t<is_function<Up>::value, add_pointer_t<Up>, remove_cv_t<Up>>>;
};

template <typename Tp>
struct decay
{
private:
    using Up = remove_reference_t<Tp>;

public:
    using type = typename DecayImpl<Up, is_referenceable<Up>::value>::type;
};

template <typename Tp>
using decay_t = typename decay<Tp>::type;

}
}

#endif // IMPL_STD_ASCENDC_STD_DECAY_IMPL__H
