/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. 2025
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file utility.h
 * \brief
 */
#ifndef LIB_STD_ASCENDC_STD_UTILITY__H
#define LIB_STD_ASCENDC_STD_UTILITY__H

#include "../../impl/std/utility/declval.h"
#include "../../impl/std/utility/forward.h"
#include "../../impl/std/utility/move.h"
#include "../../impl/std/utility/integer_sequence.h"


namespace AscendC {
namespace Std {
template <size_t... Idx>
using index_sequence = IntegerSequence<size_t, Idx...>;
template <size_t N>
using make_index_sequence = MakeIntegerSequence<size_t, N>;
}
}
#endif // LIB_STD_ASCENDC_STD_UTILITY__H
