/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef __INC_METADEF_NODE_UTILS_EX_H
#define __INC_METADEF_NODE_UTILS_EX_H

#include "graph/node.h"
#include "graph/op_desc.h"

namespace ge {
class NodeUtilsEx {
 public:
  // Detach from Node
  static graphStatus Verify(const NodePtr &node);
  static graphStatus InferShapeAndType(const NodePtr &node);
  static graphStatus InferOriginFormat(const NodePtr &node);
  // Detach from NodeUtils
  static ConstNodePtr GetNodeFromOperator(const Operator &op);
  static graphStatus SetNodeToOperator(Operator &op, const ConstNodePtr &node);
 private:
  static graphStatus IsInputsValid(const NodePtr &node);
};
} // namespace ge
#endif // __INC_METADEF_NODE_UTILS_EX_H
