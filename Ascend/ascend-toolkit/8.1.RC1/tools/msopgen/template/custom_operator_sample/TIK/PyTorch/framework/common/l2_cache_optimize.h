/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_FRAMEWORK_COMMON_L2_CACHE_OPTIMIZE_H_
#define INC_FRAMEWORK_COMMON_L2_CACHE_OPTIMIZE_H_

#include <cstdint>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/compute_graph.h"

namespace ge {
// Size of RC memory alignment, 2M
constexpr size_t ALIGN_SIZE = 2097152U;

constexpr uint32_t RC_VALUE_DEFAULT = 1U;
constexpr uint32_t RC_VALUE_MAX = 32U;

}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_L2_CACHE_OPTIMIZE_H_
