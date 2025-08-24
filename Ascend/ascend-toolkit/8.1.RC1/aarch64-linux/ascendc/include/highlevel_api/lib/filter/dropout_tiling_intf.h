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
 * \file dropout_tiling_intf.h
 * \brief
 */
#ifndef API_TILING_DROPOUT_TILING_INTF_H
#define API_TILING_DROPOUT_TILING_INTF_H
#include "dropout/dropout_tiling.h"

namespace AscendC {
[[deprecated(__FILE__ " is deprecated, please use dropout_tiling.h instead!")]]
    typedef void UsingDeprecatedHeader;
using LibTilingDropoutTilingInterface = UsingDeprecatedHeader;
} // namespace AscendC
#endif // API_TILING_DROPOUT_TILING_INTF_H