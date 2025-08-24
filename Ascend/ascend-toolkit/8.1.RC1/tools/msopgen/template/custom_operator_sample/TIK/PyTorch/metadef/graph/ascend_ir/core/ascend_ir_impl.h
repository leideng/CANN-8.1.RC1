/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GRAPH_ASCEND_IR_IMPL_H
#define GRAPH_ASCEND_IR_IMPL_H

#include <string>
#include "attr_store.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/anchor.h"
#include "graph/utils/op_desc_utils.h"
#include "common/ge_common/debug/ge_log.h"
#include "external/graph/operator.h"

namespace ge {
namespace ascir {
namespace cg {
class CodeGenUtils;
}
}
class AscGraphImpl {
  friend class AscGraph;
  friend class AscGraphUtils;
  friend class ascir::cg::CodeGenUtils;
 public:
  explicit AscGraphImpl(const char *name);

  Axis *FindAxis(const int64_t axis_id);

  void SetTilingKey(const uint32_t tiling_key);

  int64_t GetTilingKey() const;

  AscNodePtr AddNode(ge::Operator &op);

  Expression CreateSizeVar(const int64_t value);

  AxisPtr CreateAxis(const std::string &name, Axis::Type type, const ge::Expression &size,
                     const std::vector<int64_t> &from, const int64_t split_peer = 0UL);

  void CreateSizeVar(const Expression &expression);

  Expression CreateSizeVar(const std::string &name);

  std::pair<AxisPtr, AxisPtr> BlockSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                         const std::string &inner_axis_name);

  std::pair<AxisPtr, AxisPtr> TileSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                        const std::string &inner_axis_name);

  AxisPtr MergeAxis(const std::vector<int64_t> &axis_ids, const std::string &merge_axis_name);

  void ApplySplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id);

  void ApplyMerge(const AscNodePtr &node, const int64_t merged_axis_id);

  void ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original);

  void ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id);

  void ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original);

  void ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id);

  void ApplyReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);

  void ApplySchedAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);

  void ApplyTensorAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);

  bool TryApplyAxisReplace(const AscNodePtr &node, const Axis &src, const Axis &dst);

  AscNodePtr FindNode(const char *name) const;

  std::vector<AxisPtr> GetAllAxis() const;

  std::vector<SizeVarPtr> GetAllSizeVar() const;

  TransInfoRoadOfGraph GetAllAxisTransInfo() const;

  AscNodeVisitor GetAllNodes() const;

  AscNodeVisitor GetInputNodes() const;

  std::string GetName() const;

  AscOpOutput CreateContiguousData(const char *name,
                                   const ge::DataType &dt,
                                   const std::vector<ge::Axis> &axes,
                                   const std::vector<std::vector<int64_t>> &axis_continuous_map,
                                   const ge::Format &format);

  AscOpOutput CreateContiguousOut(const char *name,
                                   const ge::DataType &dt,
                                   const std::vector<ge::Axis> &axes,
                                   const ge::Format &format);

  void SortByExecOrder();

  const ComputeGraphPtr GetComputeGraph() const;

  void CopyFrom(const ge::AscGraph &src_graph, ge::AscGraph &dst_graph);
 private:
  std::pair<AxisPtr, AxisPtr> DoSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                      const std::string &inner_axis_name, const bool is_tile_split);

  void DoApplySplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id, const int64_t original_id);

  void DoApplyMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original);

  void DoApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id,
                              const std::vector<int64_t> &original);

  void DoApplyTensorAxisSplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id,
                              const int64_t original_id);

  void DoApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original);

  void DoApplySchedAxisSplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id,
                             const int64_t original_id);

  void DoApplySchedAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);

  void DoApplyTensorAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);

  void DoCopyAscGraphAttr(const AscGraph &src_asc_graph, AscGraph &dst_asc_graph);
  static void DoCopyAscGraphAttrImpl(const ComputeGraphPtr &src_compute_graph, const ComputeGraphPtr &dst_compute_graph);

  void DoCopyAscNodeAndRelink(const AscGraph &src_asc_graph, AscGraph &dst_asc_graph);

  void DoCopyAscNodeTensorAttr(const AscNodePtr &src_node, AscNodePtr &dst_node);

  AscGraphAttr *GetOrCreateGraphAttrsGroup();
  const AscGraphAttr *GetOrCreateGraphAttrsGroup() const;
 private:
  ComputeGraphPtr compute_graph_;
};
using AscGraphImplPtr = std::shared_ptr<AscGraphImpl>;
}  // namespace ge

#endif  // GRAPH_ASCEND_IR_IMPL_H
