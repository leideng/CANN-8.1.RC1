/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_GRAPH_SHAPE_REFINER_H_
#define INC_GRAPH_SHAPE_REFINER_H_

#include <string>
#include "external/graph/inference_context.h"

#include "external/graph/ge_error_codes.h"
#include "graph/node.h"
#include "graph/resource_context_mgr.h"

namespace ge {
// ShapeRefiner performs shape inference for compute graphs
class ShapeRefiner {
 public:
  static graphStatus InferShapeAndType(const ConstNodePtr &node, Operator &op, const bool before_subgraph);
  static graphStatus InferShapeAndType(const NodePtr &node, const bool before_subgraph);
  static graphStatus InferShapeAndType(const NodePtr &node);
  static graphStatus InferShapeAndType(const ConstNodePtr &node, Operator &op);
  static graphStatus DoInferShapeAndTypeForRunning(const ConstNodePtr &node, Operator &op, const bool before_subgraph);
  static graphStatus InferShapeAndTypeForRunning(const NodePtr &node, Operator &op, const bool before_subgraph);
  static void ClearContextMap();
  static graphStatus CreateInferenceContext(const NodePtr &node,
                                            InferenceContextPtr &inference_context);
  static graphStatus CreateInferenceContext(const NodePtr &node,
                                            ResourceContextMgr *const resource_context_mgr,
                                            InferenceContextPtr &inference_context);
  static void PushToContextMap(const NodePtr &node, const InferenceContextPtr &inference_context);

 private:
  static void PrintInOutTensorShape(const ge::NodePtr &node, const std::string &phase);
  static graphStatus GetRealInNodesAndIndex(NodePtr &input_node, int32_t &output_idx,
                                            std::map<NodePtr, int32_t> &nodes_idx);
  static graphStatus PostProcessAfterInfershape(const NodePtr &node, const Operator &op, const bool is_unknown_graph);
  static graphStatus UpdateInputOutputDesc(const NodePtr &node);
};
}  // namespace ge
#endif  // INC_GRAPH_SHAPE_REFINER_H_
