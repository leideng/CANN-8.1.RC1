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
 * \file kernel_operator_deepnorm_intf.h
 * \brief
 */
#ifndef LIB_NORMALIZATION_KERNEL_OPERATOR_DEEPNORM_INTF_H
#define LIB_NORMALIZATION_KERNEL_OPERATOR_DEEPNORM_INTF_H
#include "deepnorm.h"

namespace AscendC {
[[deprecated(__FILE__ " is deprecated, please use deepnorm.h instead!")]] typedef void DeepNormDeprecatedHeader;
using LibDeepNormInterface = DeepNormDeprecatedHeader;
} // namespace AscendC
#endif // LIB_NORMALIZATION_KERNEL_OPERATOR_DEEPNORM_INTF_H
