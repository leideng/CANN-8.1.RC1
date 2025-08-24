/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_operator_lgamma_intf.h
 * \brief
 */
#ifndef LIB_MATH_KERNEL_OPERATOR_LGAMMA_INTERFACE_H
#define LIB_MATH_KERNEL_OPERATOR_LGAMMA_INTERFACE_H
#include "lgamma.h"

namespace AscendC {
[[deprecated(__FILE__ " is deprecated, please use lgamma.h instead!")]] typedef void LgammaDeprecatedHeader;
using LibLgammaInterface = LgammaDeprecatedHeader;
}  // namespace AscendC
#endif  // LIB_MATH_KERNEL_OPERATOR_LGAMMA_INTERFACE_H
