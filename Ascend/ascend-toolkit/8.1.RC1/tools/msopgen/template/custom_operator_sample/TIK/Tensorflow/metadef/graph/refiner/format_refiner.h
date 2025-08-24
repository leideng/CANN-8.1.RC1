/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef COMMON_GRAPH_FORMAT_REFINER_H_
#define COMMON_GRAPH_FORMAT_REFINER_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define METADEF_FUNC_VISIBILITY _declspec(dllexport)
#else
#define METADEF_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define METADEF_FUNC_VISIBILITY
#else
#define METADEF_FUNC_VISIBILITY __attribute__((visibility("hidden")))
#endif
#endif

#include <deque>
#include <string>
#include <unordered_map>
#include <vector>
#include "graph/compute_graph.h"
#include "graph/types.h"
#include "graph/ge_error_codes.h"

namespace ge {
// ShapeRefiner performs shape inference for compute graphs
class METADEF_FUNC_VISIBILITY FormatRefiner {
 public:
  static graphStatus InferOrigineFormat(const ge::ComputeGraphPtr &graph);

 private:
  static graphStatus RefreshConstantOutProcess(const ComputeGraphPtr &com_graph, const OpDescPtr &op_desc);
  static graphStatus GetAnchorPoints(const ge::ComputeGraphPtr &com_graph, std::vector<ge::NodePtr> &anchor_points,
                                     std::vector<ge::NodePtr> &anchor_data_nodes);
  static graphStatus AnchorProcess(const ge::NodePtr &anchor_node);
  static void RefreshOriginFormatOfAnchor(const std::vector<ge::NodePtr> &anchor_points);
  static graphStatus BackInferProcess(std::deque<ge::NodePtr> &nodes, const ge::NodePtr &node);
  static graphStatus ForwardInferProcess(std::deque<ge::NodePtr> &nodes, const ge::NodePtr &node);
  static graphStatus DataNodeFormatProcess(const ComputeGraphPtr &graph,
                                           const std::vector<ge::NodePtr> &anchor_data_nodes,
                                           const ge::Format data_format);
  static bool IsGraphInferred(const ComputeGraphPtr &graph);
};
}  // namespace ge
#endif  // COMMON_GRAPH_FORMAT_REFINER_H_
