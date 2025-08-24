/* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "graph/ascend_ir/utils/asc_tensor_utils.h"

namespace ge {
namespace ascir {
bool AscTensorUtils::IsConstTensor(const AscTensor &t) {
  const auto node = t.anchor.GetOwnerNodeBarePtr();
  GE_ASSERT_NOTNULL(node);
  return node->GetType() == "Constant" || node->GetType() == "IndexExpr" || node->GetType() == "Scalar";
}
Node *AscTensorUtils::GetOwner(const AscTensor &t) {
  return t.anchor.GetOwnerNodeBarePtr();
}
int32_t AscTensorUtils::Index(const AscTensor &t) {
  return t.anchor.GetIdx();
}
}  // namespace ascir
}  // namespace ge
