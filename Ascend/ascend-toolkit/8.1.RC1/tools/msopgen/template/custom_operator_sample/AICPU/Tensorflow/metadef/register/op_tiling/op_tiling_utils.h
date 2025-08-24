/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef REGISTER_OP_TILING_OP_TILING_UTILS_H_
#define REGISTER_OP_TILING_OP_TILING_UTILS_H_

#include <vector>
#include <nlohmann/json.hpp>
#include "graph/op_desc.h"
#include "graph/debug/ge_log.h"

namespace optiling {
void ReplaceEmptyShapeOfTensorDesc(const ge::OpDescPtr &op_desc, std::vector<int32_t> &indexes);
void RecoveryEmptyShapeOfTensorDesc(const ge::OpDescPtr &op_desc, const std::vector<int32_t> &indexes);

#define OP_TILING_MAKE_SHARED(exec_expr0, exec_expr1) \
  do {                                                \
    try {                                             \
      exec_expr0;                                     \
    } catch (...) {                                   \
      GE_LOGE("Make shared failed");                  \
      exec_expr1;                                     \
    }                                                 \
  } while (0)

}  // namespace optiling
#endif  // REGISTER_OP_TILING_OP_TILING_UTILS_H_
