/* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_CXX_ASC_TENSOR_UTILS_H
#define METADEF_CXX_ASC_TENSOR_UTILS_H

#include "ascend_ir/ascend_ir_core/ascend_ir.h"

namespace ge {
namespace ascir {
class AscTensorUtils {
 public:
  static bool IsConstTensor(const AscTensor &t);
  static Node *GetOwner(const AscTensor &t);
  static int32_t Index(const AscTensor &t);
};
}
}  // namespace ge

#endif  // METADEF_CXX_ASC_TENSOR_UTILS_H
