/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_common_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_COMMON_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_COMMON_IMPL_H

#include "softmax_common_impl/softmax_common_broadcast.h"
#include "softmax_common_impl/softmax_common_nd_reduce.h"
#include "../common/softmax_common_nz_reduce.h"

namespace AscendC {
constexpr RoundMode FLOAT2HALF_ROUND_MODE = RoundMode::CAST_ROUND;

};
#endif // IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_COMMON_IMPL_H