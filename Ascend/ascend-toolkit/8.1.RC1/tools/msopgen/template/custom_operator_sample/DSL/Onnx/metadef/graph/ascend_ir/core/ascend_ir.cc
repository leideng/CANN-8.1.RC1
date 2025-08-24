/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/


#include "inc/graph/ascend_ir/ascend_ir_core/ascend_ir.h"
#include "ascend_ir_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/cg_utils.h"
#include "graph/debug/ge_log.h"
#include "graph/debug/ge_op_types.h"
#include "graph/ascend_ir/utils/asc_tensor_utils.h"
#include "graph/ascend_ir/utils/asc_graph_utils.h"
#include "expression/const_values.h"

#define CHECK_FALSE_RETURN_FALSE(expr, err_log, ...)                                                                   \
  do {                                                                                                                 \
    if (!(expr)) {                                                                                                     \
      GE_LOGE(err_log, ##__VA_ARGS__);                                                                                 \
      return false;                                                                                                    \
    }                                                                                                                  \
  } while (false)
namespace ge {
namespace {
constexpr int32_t kDefaultAlignVal = 1;
constexpr uint32_t kMinMergeAxisFromSize = 2U;
const std::vector<std::vector<int64_t>> kOneAxisContinuousInfo = {{0, -1}};
const char *const kAscData = ge::DATA;
const char *const kAscOutput = "Output";
}

// TTODO ascend attr will be split into asc_attr_group
std::unique_ptr<AttrGroupsBase> AscGraphAttr::Clone() {
  auto ptr = ComGraphMakeUnique<AscGraphAttr>(*this);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(ptr);
  return ptr;
}

std::unique_ptr<AttrGroupsBase> AscNodeAttr::Clone() {
  auto ptr = ComGraphMakeUnique<AscNodeAttr>(*this);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(ptr);
  return ptr;
}

AscNodeAttr &AscNodeAttr::CreateImpl(ge::Operator &op) {
  auto opdesc = ge::OpDescUtils::GetOpDescFromOperator(op).get();
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(opdesc);
  ASCIR_ASSERT_TRUE(opdesc->GetAttrsGroup<AscNodeAttr>() == nullptr);

  auto attr_group = opdesc->GetOrCreateAttrsGroup<AscNodeAttr>();
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(attr_group);
  return *attr_group;
}

AscNodeAttr &AscNodeAttr::operator=(const AscNodeAttr &other) {
  if (this == &other) {
    return *this;
  }
  name = other.name;
  type = other.type;
  sched = other.sched;
  api = other.api;
  tmp_buffers = other.tmp_buffers;

  if (other.ir_attr) {
    ir_attr = other.ir_attr->Clone();
  } else {
    ir_attr.reset();
  }
  return *this;
}

AscNodeAttr &AscNodeAttr::Create(Operator &op) {
  return CreateImpl(op);
}

AscTensorAttr &AscTensorAttr::GetTensorAttr(ge::Operator *op, const uint32_t index) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(op);
  const auto desc = ge::OpDescUtils::GetOpDescFromOperator(*op);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(desc);
  const auto tensor_desc = desc->MutableOutputDesc(index);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(tensor_desc);
  const auto attr_group = tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(attr_group);
  attr_group->dtype.tensor_desc_ = tensor_desc.get();
  return *attr_group;
}

AscTensorAttr *AscTensorAttr::GetTensorAttrPtr(ge::Operator *op, const uint32_t index) {
  GE_ASSERT_NOTNULL(op);
  const auto desc = ge::OpDescUtils::GetOpDescFromOperator(*op);
  GE_ASSERT_NOTNULL(desc);
  auto tensor = desc->MutableOutputDesc(index);
  if (tensor == nullptr) {
    return nullptr;
  }
  const auto attr_group = tensor->GetOrCreateAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(attr_group);
  attr_group->dtype.tensor_desc_ = tensor.get();
  return attr_group;
}

AscTensorAttr &AscTensorAttr::GetTensorAttr(const OutDataAnchor &output) {
  const auto node = output.GetOwnerNodeBarePtr();
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  const auto op_desc = node->GetOpDescBarePtr();
  const auto tensor_desc = op_desc->MutableOutputDesc(output.GetIdx());
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(tensor_desc);
  const auto attr_group = tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(attr_group);
  attr_group->dtype.tensor_desc_ = tensor_desc.get();
  return *attr_group;
}

std::unique_ptr<AttrGroupsBase> AscTensorAttr::Clone() {
  auto ptr = ComGraphMakeUnique<AscTensorAttr>(*this);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(ptr);
  return ptr;
}

void AscNodeOutputs::Init() {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node_);
  for (const auto &output : node_->GetAllOutDataAnchorsPtr()) {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(output);
    tensors_.emplace_back(AscTensor(*output));
  }
}

AscTensor &AscNodeOutputs::operator[](uint32_t index) {
  if (tensors_.empty()) {
    Init();
  }
  CHECK_BOOL_WITH_THROW_EXCEPTION(GRAPH_PARAM_INVALID, index < tensors_.size(), "index = %u but tensors_.size() = %zu",
                                  index, tensors_.size());
  return tensors_[index];
}

std::vector<AscTensor *> AscNodeOutputs::operator()() {
  if (tensors_.empty()) {
    Init();
  }
  if (tensors_.empty()) {
    return {};
  }
  std::vector<AscTensor *> tensors;
  for (auto &tensor : tensors_) {
    tensors.push_back(&tensor);
  }
  return tensors;
}

void AscNodeInputs::Init() {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node_);
  std::vector<AscTensor> tmp_tensors;
  for (const auto &in_anchor : node_->GetAllInDataAnchorsPtr()) {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(in_anchor);
    const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      GELOGD("node[%s, %s] link [%d] are not ready", node_->GetNamePtr(), node_->GetTypePtr(), in_anchor->GetIdx());
      continue;
    }
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(peer_out_anchor);
    tmp_tensors.emplace_back(AscTensor(*peer_out_anchor));
  }
  tensors_ = std::move(tmp_tensors);
}

// make sure ascend graph is fixed, if index 1 is first linked, tensors_ 0 means index 1, that may cause bug
AscTensor &AscNodeInputs::operator[](uint32_t index) {
  // as not all input is ready at the same time, must call Init on every function call
  Init();
  CHECK_BOOL_WITH_THROW_EXCEPTION(GRAPH_PARAM_INVALID, index < tensors_.size());
  return tensors_[index];
}

std::vector<AscTensor *> AscNodeInputs::operator()() {
  // as not all input is ready at the same time, must call Init on every function call
  Init();
  if (tensors_.empty()) {
    return {};
  }
  const auto node = ascir::AscTensorUtils::GetOwner(tensors_[0]);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  auto op_desc = node->GetOpDesc();
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_desc);
  std::vector<AscTensor *> tensors;
  for (auto &tensor : tensors_) {
    tensors.emplace_back(&tensor);
  }
  return tensors;
}

uint32_t AscNodeInputs::Size() {
  // as not all input is ready at the same time, must call Init on every function call
  Init();
  return tensors_.size();
}

// 此处op_desc和GetOrCreateAttrsGroup的返回值未判空,内部构造AscNode前已判空
// 资料需注明不允许外部用户构造AscNode
AscNode::AscNode(const OpDescPtr &op_desc, const ComputeGraphPtr &compute_graph) :
    Node(op_desc, compute_graph), inputs(this), outputs(this),
    attr(*(op_desc->GetOrCreateAttrsGroup<AscNodeAttr>())) {
  if (op_desc != nullptr) {
    attr.name = op_desc->GetName();
    attr.type = op_desc->GetType();
  }
}

AscNodeIter::AscNodeIter(ge::ComputeGraph::Vistor<ge::NodePtr>::Iterator &&iter) : impl_(iter) {}

AscNodeIter &AscNodeIter::operator++() {
  impl_++;
  return *this;
}

AscNodePtr AscNodeIter::operator*() {
  auto ptr = *impl_;
  return std::dynamic_pointer_cast<AscNode>(ptr);
}

bool AscNodeIter::operator!=(const AscNodeIter &other) const {
  return impl_ != other.impl_;
}

AscNodeVisitor::AscNodeVisitor(ge::ComputeGraph::Vistor<ge::NodePtr> &&visitor)
    : impl_(visitor) {}

AscNodeIter AscNodeVisitor::begin() {
  return AscNodeIter(impl_.begin());
}

AscNodeIter AscNodeVisitor::end() {
  return AscNodeIter(impl_.end());
}

AscGraphImpl::AscGraphImpl(const char *name) :
  compute_graph_(ComGraphMakeShared<ComputeGraph>(name)) {}

std::string AscGraphImpl::GetName() const {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(compute_graph_);
  return compute_graph_->GetName();
}


void AscGraphImpl::SetTilingKey(const uint32_t tiling_key) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  graph_attr_group_ptr->tiling_key = static_cast<int64_t>(tiling_key);
}

int64_t AscGraphImpl::GetTilingKey() const{
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  return graph_attr_group_ptr->tiling_key;
}

AscNodePtr AscGraphImpl::AddNode(ge::Operator &op) {
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_desc);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(compute_graph_);
  AscNodePtr asc_node = std::make_shared<AscNode>(op_desc, compute_graph_);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(asc_node);
  const auto init_ret = asc_node->Init();
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, init_ret == GRAPH_SUCCESS);
  ConstNodePtr const_node = asc_node;
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED,
                                  ge::NodeUtilsEx::SetNodeToOperator(op, const_node) == GRAPH_SUCCESS);
  auto node = compute_graph_->AddNode(asc_node);
  auto new_node = std::dynamic_pointer_cast<AscNode>(node);
  // update
  (void) new_node->inputs();
  (void) new_node->outputs();
  return new_node;
}

AscNodePtr AscGraphImpl::FindNode(const char *name) const{
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(compute_graph_);
  auto node = compute_graph_->FindNode(name);
  auto dst_node = std::dynamic_pointer_cast<AscNode>(node);
  return dst_node;
}

AscNodeVisitor AscGraphImpl::GetAllNodes() const{
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(compute_graph_);
  return AscNodeVisitor(compute_graph_->GetAllNodes());
}

AscNodeVisitor AscGraphImpl::GetInputNodes() const{
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(compute_graph_);
  return AscNodeVisitor(compute_graph_->GetInputNodes());
}

ge::Expression AscGraphImpl::CreateSizeVar(const int64_t value) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  const auto expr = Symbol(value);
  const auto size_var = ComGraphMakeShared<SizeVar>(expr);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(size_var);
  graph_attr_group_ptr->size_vars.push_back(size_var);
  return graph_attr_group_ptr->size_vars.back()->expr;
}

ge::Expression AscGraphImpl::CreateSizeVar(const std::string &name) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  const auto expr = Symbol(name.c_str());
  const auto size_var = ComGraphMakeShared<SizeVar>(expr);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(size_var);
  graph_attr_group_ptr->size_vars.push_back(size_var);
  return graph_attr_group_ptr->size_vars.back()->expr;
}

AxisPtr AscGraphImpl::CreateAxis(const std::string &name, Axis::Type type,
                                 const Expression &size, const std::vector<int64_t> &from, const int64_t split_peer) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  auto axis = ComGraphMakeShared<Axis>();
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(axis);
  axis->type = type;
  axis->name = name;
  axis->size = size;
  axis->from = from;
  axis->align = kDefaultAlignVal;
  axis->split_pair_other_id = split_peer;
  axis->allow_oversize_axis = false;
  axis->allow_unaligned_tail = true;
  axis->id = static_cast<int64_t>(graph_attr_group_ptr->axis.size());

  graph_attr_group_ptr->axis.push_back(std::move(axis));

  return graph_attr_group_ptr->axis.back();
}

std::vector<AxisPtr> AscGraphImpl::GetAllAxis() const{
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  return graph_attr_group_ptr->axis;
}

std::vector<SizeVarPtr> AscGraphImpl::GetAllSizeVar() const{
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  return graph_attr_group_ptr->size_vars;
}

TransInfoRoadOfGraph AscGraphImpl::GetAllAxisTransInfo() const {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  return graph_attr_group_ptr->trans_info_road;
}

Axis *AscGraphImpl::FindAxis(const int64_t axis_id) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  if (axis_id < 0 || axis_id > static_cast<int64_t>(graph_attr_group_ptr->axis.size())) {
    return nullptr;
  }
  return graph_attr_group_ptr->axis[axis_id].get();
}

std::pair<AxisPtr, AxisPtr> AscGraphImpl::DoSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                                  const std::string &inner_axis_name, const bool is_tile_split) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  const auto &axis = graph_attr_group_ptr->axis;
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, (axis_id >= 0) && (static_cast<size_t>(axis_id) < axis.size()));

  const auto &single_axis = *axis[axis_id];
  const std::string inner_suffix = is_tile_split ? "t" : "b";
  const std::string outer_suffix = is_tile_split ? "T" : "B";
  std::string actual_inner_axis_name = inner_axis_name;
  if (actual_inner_axis_name.empty()) {
    actual_inner_axis_name = single_axis.name + inner_suffix;
  }
  std::string actual_outer_axis_name = outer_axis_name;
  if (actual_outer_axis_name.empty()) {
    actual_outer_axis_name = single_axis.name + outer_suffix;
  }
  const auto inner_size = CreateSizeVar(actual_inner_axis_name + "_size");
  const auto outer_size = ge::sym::Ceiling(single_axis.size / inner_size);
  Axis::Type inner_type = is_tile_split ? Axis::kAxisTypeTileInner : Axis::kAxisTypeBlockInner;
  Axis::Type outer_type = is_tile_split ? Axis::kAxisTypeTileOuter : Axis::kAxisTypeBlockOuter;
  int64_t outter_id = static_cast<int64_t>(graph_attr_group_ptr->axis.size());
  int64_t inner_id = outter_id + 1;
  AxisPtr outer = CreateAxis(actual_outer_axis_name, outer_type, outer_size, {axis_id}, inner_id);
  AxisPtr inner = CreateAxis(actual_inner_axis_name, inner_type, inner_size, {axis_id}, outter_id);
  graph_attr_group_ptr->trans_info_road.push_back({TransType::kSplit, {axis[axis_id]}, {outer, inner}});
  return {outer, inner};
}

std::pair<AxisPtr, AxisPtr> AscGraphImpl::BlockSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                                     const std::string &inner_axis_name) {
  return DoSplit(axis_id, outer_axis_name, inner_axis_name, false);
}

std::pair<AxisPtr, AxisPtr> AscGraphImpl::TileSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                                    const std::string &inner_axis_name) {
  return DoSplit(axis_id, outer_axis_name, inner_axis_name, true);
}

AxisPtr AscGraphImpl::MergeAxis(const std::vector<int64_t> &axis_ids, const std::string &merge_axis_name) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  const auto &axis = graph_attr_group_ptr->axis;
  std::string name;
  Expression size = sym::kSymbolOne;
  std::vector<int64_t> from_axis_ids;
  std::vector<AxisPtr> from_axis;
  for (const auto &axis_id : axis_ids) {
    CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, (axis_id >= 0) && (static_cast<size_t>(axis_id) < axis.size()));
    from_axis.push_back(axis[axis_id]);
    name += axis[axis_id]->name;
    size = size * axis[axis_id]->size;
    from_axis_ids.push_back(axis_id);
  }
  name = merge_axis_name.empty() ? name : merge_axis_name;
  AxisPtr merge_axis = CreateAxis(name, Axis::kAxisTypeMerged, size, from_axis_ids);
  graph_attr_group_ptr->trans_info_road.push_back({TransType::kMerge, from_axis, {merge_axis}});
  return merge_axis;
}

void AscGraphImpl::DoApplySplit(const AscNodePtr &node, const int64_t outter_id, int64_t inner_id, int64_t original) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  DoApplySchedAxisSplit(node, outter_id, inner_id, original);
  DoApplyTensorAxisSplit(node, outter_id, inner_id, original);
}

void AscGraphImpl::DoApplyTensorAxisSplit(const AscNodePtr &node, const int64_t outter_id,
                                          const int64_t inner_id, const int64_t original_id) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  const auto &all_axis = graph_attr_group_ptr->axis;
  // check inner_axis before
  const Expression &split_size = all_axis[inner_id]->size;
  for (uint32_t i = 0; i < node->GetAllOutDataAnchorsSize(); i++) {
    const auto &result =
        AxisUtils::SplitView({node->outputs[i].attr.axis,
                              node->outputs[i].attr.repeats, node->outputs[i].attr.strides},
                             split_size,
                             outter_id,
                             inner_id,
                             original_id);
    node->outputs[i].attr.axis = result.axis_ids;
    node->outputs[i].attr.repeats = result.repeats;
    node->outputs[i].attr.strides = result.strides;
  }
}

void AscGraphImpl::DoApplySchedAxisSplit(const AscNodePtr &node, const int64_t outter_id,
                                         const int64_t inner_id, const int64_t original_id) {
  std::vector<int64_t> new_node_attr_axis;
  const auto &node_axis = node->attr.sched.axis;
  for (auto &node_axis_id : node_axis) {
    if (node_axis_id == original_id) {
      new_node_attr_axis.push_back(outter_id);
      new_node_attr_axis.push_back(inner_id);
    } else {
      new_node_attr_axis.push_back(node_axis_id);
    }
  }
  node->attr.sched.axis = new_node_attr_axis;
}

void AscGraphImpl::ApplySplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  const auto &all_axis = graph_attr_group_ptr->axis;
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED,
    (outter_id >= 0) && (outter_id < static_cast<int64_t>(all_axis.size())) &&
    (inner_id >= 0) && (inner_id < static_cast<int64_t>(all_axis.size())));
  const auto &out_axis = *all_axis[outter_id];
  const auto &in_axis = *all_axis[inner_id];
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, (out_axis.type == Axis::kAxisTypeBlockOuter &&
    in_axis.type == Axis::kAxisTypeBlockInner) ||
    (out_axis.type == Axis::kAxisTypeTileOuter && in_axis.type == Axis::kAxisTypeTileInner));
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED,
    (out_axis.from.size() == 1U) && (in_axis.from.size() == 1U) && (out_axis.from[0] == in_axis.from[0]));
  DoApplySplit(node, outter_id, inner_id, out_axis.from[0]);
}

void AscGraphImpl::DoApplyMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  DoApplySchedAxisMerge(node, merged_axis_id, original);
  DoApplyTensorAxisMerge(node, merged_axis_id, original);
}

void AscGraphImpl::DoApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id,
                                         const std::vector<int64_t> &original) {
  std::vector<int64_t> new_node_attr_axis;
  auto cur_iter = original.begin();
  for (const auto axis_id : node->attr.sched.axis) {
    if (cur_iter != original.end() && axis_id == *cur_iter) {
      cur_iter++;
      if (cur_iter == original.end()) {
        new_node_attr_axis.push_back(merged_axis_id);
      }
    } else {
      new_node_attr_axis.push_back(axis_id);
    }
  }
  ASCIR_ASSERT_TRUE(
      cur_iter == original.begin() || cur_iter == original.end(),
      "node {%s} has sched.axis %s but origin is %s",
      node->GetNamePtr(),
      ViewMemberToString(node->attr.sched.axis).c_str(),
      ViewMemberToString(original).c_str());
  node->attr.sched.axis = new_node_attr_axis;
}

void AscGraphImpl::DoApplySchedAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  const auto &node_axis = node->attr.sched.axis;
  for (const auto axis_id : reordered_axis) {
    const auto it = std::find(node_axis.begin(), node_axis.end(), axis_id);
    CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, it != node_axis.end(),
                                    "can not find axis_id[%ld] of reordered_axis, node[%s,%s]", axis_id,
                                    node->GetNamePtr(), node->GetTypePtr());
  }
  node->attr.sched.axis = reordered_axis;
}

void AscGraphImpl::DoApplyTensorAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  const auto &node_axis = node->attr.sched.axis;
  for (const auto axis_id : reordered_axis) {
    const auto it = std::find(node_axis.begin(), node_axis.end(), axis_id);
    CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, it != node_axis.end());
  }
  for (const auto output_ptr : node->outputs()) {
    auto &output = *output_ptr;
    std::vector<int64_t> new_axis;
    std::vector<Expression> new_repeat;
    std::vector<Expression> new_strides;
    auto output_axis = output.attr.axis;
    for (const auto axis_id : reordered_axis) {
      const auto it = std::find(output_axis.begin(), output_axis.end(), axis_id);
      if (it == output_axis.end()) {
        continue;
      }
      const auto pos = std::distance(output_axis.begin(), it);
      new_axis.push_back(output_axis[pos]);
      new_repeat.push_back(output.attr.repeats[pos]);
      new_strides.push_back(output.attr.strides[pos]);
    }
    output.attr.axis = new_axis;
    output.attr.repeats = new_repeat;
    output.attr.strides = new_strides;
  }
}
void AscGraphImpl::DoCopyAscGraphAttr(const AscGraph &src_asc_graph, AscGraph &dst_asc_graph) {
  return DoCopyAscGraphAttrImpl(AscGraphUtils::GetComputeGraph(src_asc_graph),
                                AscGraphUtils::GetComputeGraph(dst_asc_graph));
}

void AscGraphImpl::DoCopyAscGraphAttrImpl(const ComputeGraphPtr &src_compute_graph,
                                          const ComputeGraphPtr &dst_compute_graph) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(src_compute_graph);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(dst_compute_graph);
  const auto dst_graph_attr = dst_compute_graph->GetOrCreateAttrsGroup<AscGraphAttr>();
  const auto src_graph_attr = src_compute_graph->GetOrCreateAttrsGroup<AscGraphAttr>();
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(dst_graph_attr);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(src_graph_attr);

  dst_graph_attr->tiling_key = src_graph_attr->tiling_key;
  for (size_t i = 0U; i < src_graph_attr->axis.size(); i++) {
    auto src_axis = src_graph_attr->axis[i];
    std::shared_ptr<Axis> axis(new(std::nothrow) Axis);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(axis);
    axis->id = src_axis->id;
    axis->name = src_axis->name;
    axis->type = src_axis->type;
    axis->bind_block = src_axis->bind_block;
    axis->size = src_axis->size;
    axis->align = src_axis->align;
    axis->from = src_axis->from;
    axis->split_pair_other_id = src_axis->split_pair_other_id;
    axis->allow_oversize_axis = src_axis->allow_oversize_axis;
    axis->allow_unaligned_tail = src_axis->allow_unaligned_tail;
    dst_graph_attr->axis.push_back(std::move(axis));
  }

  for (const auto &src_sizevar : src_graph_attr->size_vars) {
    auto size_var = ComGraphMakeShared<SizeVar>(src_sizevar->expr);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(size_var);
    dst_graph_attr->size_vars.push_back(std::move(size_var));
  }
}

void AscGraphImpl::DoCopyAscNodeAndRelink(const AscGraph &src_asc_graph, AscGraph &dst_asc_graph) {
  const auto src_compute_graph = AscGraphUtils::GetComputeGraph(src_asc_graph);
  auto dst_compute_graph = AscGraphUtils::GetComputeGraph(dst_asc_graph);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(src_compute_graph);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(dst_compute_graph);
  std::unordered_map<std::string, NodePtr> all_new_nodes;
  for (const auto &src_node : src_asc_graph.GetAllNodes()) {
    const auto &op_desc = GraphUtils::CopyOpDesc(src_node->GetOpDesc(), nullptr);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_desc);
    op_desc->SetName(src_node->GetName());
    ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    auto dst_new_node = dst_asc_graph.AddNode(op);
    all_new_nodes[dst_new_node->GetName()] = std::dynamic_pointer_cast<Node>(dst_new_node);
    DoCopyAscNodeTensorAttr(src_node, dst_new_node);
  }

  for (const auto &src_node : src_compute_graph->GetAllNodes()) {
    CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, GraphUtils::RelinkGraphEdges(src_node, "", all_new_nodes) == GRAPH_SUCCESS);
  }
}

void AscGraphImpl::DoCopyAscNodeTensorAttr(const AscNodePtr &src_node, AscNodePtr &dst_node) {
  // op_desc保证非空
  auto op_desc = dst_node->GetOpDesc();
  auto dst_asc_node_attr = op_desc->GetOrCreateAttrsGroup<AscNodeAttr>();
  auto src_asc_node_attr = src_node->GetOpDesc()->GetOrCreateAttrsGroup<AscNodeAttr>();
  if (src_asc_node_attr != nullptr && dst_asc_node_attr != nullptr) {
    *dst_asc_node_attr = *src_asc_node_attr;
  }
  for (uint32_t i = 0; i < src_node->outputs().size(); i++) {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_desc->MutableOutputDesc(i));
    auto tensor_attr_group = op_desc->MutableOutputDesc(i)->GetOrCreateAttrsGroup<AscTensorAttr>();
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(tensor_attr_group);
    tensor_attr_group->dtype = src_node->outputs[i].attr.dtype;
    tensor_attr_group->axis = src_node->outputs[i].attr.axis;
    tensor_attr_group->repeats = src_node->outputs[i].attr.repeats;
    tensor_attr_group->strides = src_node->outputs[i].attr.strides;
    tensor_attr_group->vectorized_axis = src_node->outputs[i].attr.vectorized_axis;
    tensor_attr_group->vectorized_strides = src_node->outputs[i].attr.vectorized_strides;
    tensor_attr_group->mem = src_node->outputs[i].attr.mem;
    tensor_attr_group->que = src_node->outputs[i].attr.que;
    tensor_attr_group->buf = src_node->outputs[i].attr.buf;
    tensor_attr_group->opt = src_node->outputs[i].attr.opt;
  }
}

// original中的轴不连续时没法做合轴
// 判断轴是否连续 stride_i == repeat_{i+1} * stride_{i+1}
bool CheckContinuous(const AscNodePtr &node, const uint32_t tensor_index, const std::vector<int64_t> &original) {
  std::vector<ge::Expression> repeats;
  std::vector<ge::Expression> strides;
  auto cur = original.begin();
  auto axis = node->outputs[tensor_index].attr.axis;
  for (uint32_t axis_index = 0U; axis_index < axis.size(); axis_index++) {
    if (cur != original.end() && axis[axis_index] == *cur) {
      cur++;
      repeats.emplace_back(node->outputs[tensor_index].attr.repeats[axis_index]);
      strides.emplace_back(node->outputs[tensor_index].attr.strides[axis_index]);
    }
  }
  ASCIR_ASSERT_TRUE(
      cur == original.begin() || cur == original.end(),
      "node {%s}'s output[%u] has axis %s but origin is %s",
      node->GetNamePtr(), tensor_index,
      ViewMemberToString(axis).c_str(),
      ViewMemberToString(original).c_str());
  if (repeats.size() <= 1U) {
    return true;
  }
  for (uint32_t i = 0U; i < repeats.size() - 1; i++) {
    if ((strides[i]) != (repeats[i + 1] * strides[i + 1])) {
      GELOGD("strides of %u is %s but {repeats * strides} of %u is %s",
             i,
             strides[i].Str().get(),
             i + 1,
             (repeats[i + 1] * strides[i + 1]).Str().get());
      return false;
    }
  }
  return true;
}

void AscGraphImpl::DoApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id,
                                          const std::vector<int64_t> &original) {
  for (uint32_t i = 0; i < node->GetAllOutDataAnchorsSize(); i++) {
    if (!CheckContinuous(node, i, original)) {
      GELOGW("%s's [%u]th output's view is not continuous.", node->GetNamePtr(), i);
      continue;
    }
    const auto &view =
        AxisUtils::MergeView({node->outputs[i].attr.axis, node->outputs[i].attr.repeats, node->outputs[i].attr.strides},
                             merged_axis_id, original);
    node->outputs[i].attr.axis = view.axis_ids;
    node->outputs[i].attr.repeats = view.repeats;
    node->outputs[i].attr.strides = view.strides;
  }
}

void AscGraphImpl::ApplyMerge(const AscNodePtr &node, const int64_t merged_axis_id) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  const auto &all_axis = graph_attr_group_ptr->axis;
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED,
    (merged_axis_id >= 0) && (merged_axis_id < static_cast<int64_t>(all_axis.size())));
  const auto &axis = *all_axis[merged_axis_id];
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, (axis.type == Axis::kAxisTypeMerged) &&
    axis.from.size() >= kMinMergeAxisFromSize);
  DoApplyMerge(node, merged_axis_id, axis.from);
}

void AscGraphImpl::ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  const auto &all_axis = graph_attr_group_ptr->axis;
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED,
                                  (merged_axis_id >= 0) && (merged_axis_id < static_cast<int64_t>(all_axis.size())));
  const auto &axis = *all_axis[merged_axis_id];
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED,
                                  (axis.type == Axis::kAxisTypeMerged) && axis.from.size() >= kMinMergeAxisFromSize);
  DoApplyTensorAxisMerge(node, merged_axis_id, axis.from);
}

void AscGraphImpl::ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  const auto &all_axis = graph_attr_group_ptr->axis;
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED,
                                  (merged_axis_id >= 0) && (merged_axis_id < static_cast<int64_t>(all_axis.size())));
  const auto &axis = *all_axis[merged_axis_id];
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED,
                                  (axis.type == Axis::kAxisTypeMerged) && axis.from.size() >= kMinMergeAxisFromSize);
  DoApplySchedAxisMerge(node, merged_axis_id, axis.from);
}

void AscGraphImpl::ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  DoApplyTensorAxisMerge(node, merged_axis_id, original);
}

void AscGraphImpl::ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  DoApplySchedAxisMerge(node, merged_axis_id, original);
}

void AscGraphImpl::ApplyReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  DoApplySchedAxisReorder(node, reordered_axis);
  DoApplyTensorAxisReorder(node, reordered_axis);
  return;
}

void AscGraphImpl::ApplySchedAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  const auto &node_axis = node->attr.sched.axis;
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, node_axis.size() == reordered_axis.size());
  DoApplySchedAxisReorder(node, reordered_axis);
}

void AscGraphImpl::ApplyTensorAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  const auto &node_axis = node->attr.sched.axis;
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, node_axis.size() == reordered_axis.size());
  DoApplyTensorAxisReorder(node, reordered_axis);
}

bool AscGraphImpl::TryApplyAxisReplace(const AscNodePtr &node, const Axis &src, const Axis &dst) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
  std::vector<ascir::AxisId> new_axes = node->attr.sched.axis;
  bool found{false};
  for (int64_t &id : new_axes) {
    if (id == src.id) {
      id = dst.id;
      found = true;
    }
  }
  node->attr.sched.axis = new_axes;
  for (auto outputs : node->outputs()) {
    auto new_output_axes = outputs->attr.axis;
    for (auto &id : new_output_axes) {
      if (id == src.id) {
        id = dst.id;
        found = true;
      }
    }
    outputs->attr.axis = new_output_axes;
  }
  return found;
}

AscGraphAttr *AscGraphImpl::GetOrCreateGraphAttrsGroup() {
  return const_cast<AscGraphAttr *>(static_cast<const AscGraphImpl *>(this)->GetOrCreateGraphAttrsGroup());
}

const AscGraphAttr *AscGraphImpl::GetOrCreateGraphAttrsGroup() const {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(compute_graph_);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(compute_graph_->GetOrCreateAttrsGroup<AscGraphAttr>());
  return compute_graph_->GetOrCreateAttrsGroup<AscGraphAttr>();
}

AscOpOutput AscGraphImpl::CreateContiguousData(const char *name,
                                               const ge::DataType &dt,
                                               const vector<ge::Axis> &axes,
                                               const vector<std::vector<int64_t>> &axis_continuous_map,
                                               const Format &format) {
  ASCIR_ASSERT_EQ(axes.size(), axis_continuous_map.size());
  auto data_op_desc = OpDescBuilder(name, kAscData).AddOutput("y").Build();
  ASCIR_ASSERT_NOTNULL(data_op_desc);
  // TTODO need create by ascir operator, now force add output and attr
  data_op_desc->AppendIrAttrName("index");
  data_op_desc->AppendIrOutput("y", kIrOutputRequired);
  auto data_op = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(data_op_desc));
  ASCIR_ASSERT_NOTNULL(data_op);
  auto data_attr = data_op_desc->GetOrCreateAttrsGroup<AscNodeAttr>();
  ASCIR_ASSERT_NOTNULL(data_attr);
  AddNode(*data_op);
  data_op_desc->SetExtAttr(ascir::cg::RELATED_OP, data_op);
  data_attr->sched.exec_order = ascir::cg::CodeGenUtils::GenNextExecId(*data_op);
  auto data_ir_attr = ComGraphMakeUnique<AscDataIrAttrDef>();
  ASCIR_ASSERT_NOTNULL(data_ir_attr);
  ASCIR_ASSERT_GRAPH_SUCCESS(data_ir_attr->SetIndex(data_attr->sched.exec_order));
  data_attr->ir_attr = std::move(data_ir_attr);

  AscOpOutput asc_op_output(data_op.get(), 0U); // data只有一个输出
  asc_op_output.dtype = dt;
  asc_op_output.format = format; // tensor上的format
  asc_op_output.SetContiguousView(axes);
  *asc_op_output.vectorized_axis = AxisUtils::GetDefaultVectorizedAxis(*asc_op_output.axis, -1);
  return asc_op_output;
}

AscOpOutput AscGraphImpl::CreateContiguousOut(const char *name,
                                              const DataType &dt,
                                              const vector<ge::Axis> &axes,
                                              const Format &format) {
  auto out_op_desc = OpDescBuilder(name, kAscOutput).AddInput("x").AddOutput("y").Build();
  ASCIR_ASSERT_NOTNULL(out_op_desc);
  auto out_op = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(out_op_desc));
  ASCIR_ASSERT_NOTNULL(out_op);
  AddNode(*out_op);
  out_op_desc->SetExtAttr(ascir::cg::RELATED_OP, out_op);
  AscOpOutput asc_op_output(out_op.get(), 0U); // output只有一个输出
  asc_op_output.dtype = dt;
  asc_op_output.format = format;
  asc_op_output.SetContiguousView(axes);
  *asc_op_output.vectorized_axis = AxisUtils::GetDefaultVectorizedAxis(*asc_op_output.axis, -1);
  return asc_op_output;
}

void AscGraphImpl::SortByExecOrder() {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(compute_graph_);
  compute_graph_->TopologicalSorting([](const ge::NodePtr &a, const ge::NodePtr &b) {
    auto node_a = std::dynamic_pointer_cast<ge::AscNode>(a);
    auto node_b = std::dynamic_pointer_cast<ge::AscNode>(b);
    return node_a->attr.sched.exec_order < node_b->attr.sched.exec_order;
  });
}

const ComputeGraphPtr AscGraphImpl::GetComputeGraph() const {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(compute_graph_);
  return compute_graph_;
}

void AscGraphImpl::CopyFrom(const ge::AscGraph &src_graph, ge::AscGraph &dst_graph) {
  DoCopyAscGraphAttr(src_graph, dst_graph);
  DoCopyAscNodeAndRelink(src_graph, dst_graph);
}

void AscGraphImpl::CreateSizeVar(const Expression &expression) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  const auto size_var = ComGraphMakeShared<SizeVar>(expression);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(size_var);
  graph_attr_group_ptr->size_vars.push_back(size_var);
}

AscGraph::AscGraph(const char *name) :
  impl_(std::shared_ptr<AscGraphImpl>(new (std::nothrow) AscGraphImpl(name))) {}

std::string AscGraph::GetName() const {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->GetName();
}

void AscGraph::SortByExecOrder() {
  impl_->SortByExecOrder();
}

void AscGraph::CopyFrom(const ge::AscGraph &graph) {
  return impl_->CopyFrom(graph, *this);
}

void AscGraph::SetTilingKey(const uint32_t tiling_key) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  impl_->SetTilingKey(tiling_key);
}

int64_t AscGraph::GetTilingKey() const{
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->GetTilingKey();
}

Expression AscGraph::CreateSizeVar(const int64_t value) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->CreateSizeVar(value);
}

Expression AscGraph::CreateSizeVar(const std::string &name) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->CreateSizeVar(name);
}

void AscGraph::CreateSizeVar(const Expression &expression) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->CreateSizeVar(expression);
}

Axis &AscGraph::CreateAxis(const std::string &name, const Expression &size) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return *(impl_->CreateAxis(name, Axis::kAxisTypeOriginal, size, {}));
}

Axis &AscGraph::CreateAxis(const std::string &name, Axis::Type type, const Expression &size,
                           const std::vector<AxisId> &from, AxisId split_peer) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return *(impl_->CreateAxis(name, type, size, from, split_peer));
}

Axis *AscGraph::FindAxis(const int64_t axis_id) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->FindAxis(axis_id);
}

AscNodePtr AscGraph::AddNode(ge::Operator &op) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->AddNode(op);
}

AscNodePtr AscGraph::FindNode(const char *name) const {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->FindNode(name);
}

AscNode &AscGraph::Node(const char *name) const {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  auto node_ptr = impl_->FindNode(name);
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(node_ptr);
  return *node_ptr;
}

AscNodeVisitor AscGraph::GetAllNodes() const {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->GetAllNodes();
}

AscNodeVisitor AscGraph::GetInputNodes() const {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->GetInputNodes();
}

std::pair<AxisPtr, AxisPtr> AscGraph::BlockSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                                 const std::string &inner_axis_name) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  CHECK_VALID_IDENTIFIER_ALLOW_EMPTY_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, inner_axis_name);
  CHECK_VALID_IDENTIFIER_ALLOW_EMPTY_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, outer_axis_name);
  return impl_->BlockSplit(axis_id, outer_axis_name, inner_axis_name);
}

std::pair<AxisPtr, AxisPtr> AscGraph::TileSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                                const std::string &inner_axis_name) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->TileSplit(axis_id, outer_axis_name, inner_axis_name);
}

AxisPtr AscGraph::MergeAxis(const std::vector<int64_t> &axis_ids, const std::string &merge_axis_name) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->MergeAxis(axis_ids, merge_axis_name);
}

void AscGraph::ApplySplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  impl_->ApplySplit(node, outter_id, inner_id);
}

void AscGraph::ApplyMerge(const AscNodePtr &node, const int64_t merged_axis_id) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  impl_->ApplyMerge(node, merged_axis_id);
}

void AscGraph::ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  impl_->ApplySchedAxisMerge(node, merged_axis_id);
}

void AscGraph::ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  impl_->ApplyTensorAxisMerge(node, merged_axis_id);
}

void AscGraph::ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  impl_->ApplySchedAxisMerge(node, merged_axis_id, original);
}

void AscGraph::ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  impl_->ApplyTensorAxisMerge(node, merged_axis_id, original);
}

void AscGraph::ApplyReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  impl_->ApplyReorder(node, reordered_axis);
}

void AscGraph::ApplySchedAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  impl_->ApplySchedAxisReorder(node, reordered_axis);
}

void AscGraph::ApplyTensorAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  impl_->ApplyTensorAxisReorder(node, reordered_axis);
}

bool AscGraph::TryApplyAxisReplace(const AscNodePtr &node, const Axis &src, const Axis &dst) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->TryApplyAxisReplace(node, src, dst);
}

std::vector<AxisPtr> AscGraph::GetAllAxis() const{
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->GetAllAxis();
}

std::vector<SizeVarPtr> AscGraph::GetAllSizeVar() const{
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(impl_);
  return impl_->GetAllSizeVar();
}

AscGraph::~AscGraph() {
  for (const auto &node: impl_->GetAllNodes()) {
    if (node == nullptr) {
      continue;
    }
    const auto &op_desc = node->GetOpDesc();
    if (op_desc != nullptr) {
      // 打破shared ptr的循环引用
      op_desc->DelExtAttr(ascir::cg::RELATED_OP);
    }
  }
}

bool AscGraph::CheckExprValid() const {
  int32_t node_index = -1;
  for (const auto &node : GetAllNodes()) {
    node_index++;
    CHECK_FALSE_RETURN_FALSE(node != nullptr, "Node ptr is null, index[%d].", node_index);
    int32_t output_index = -1;
    for (const auto &tensor : node->outputs()) {
      output_index++;
      CHECK_FALSE_RETURN_FALSE(tensor != nullptr, "Tensor ptr is null, index[%d], node name[%s].", output_index,
                               node->GetName().c_str());
    }
  }
  return true;
}

bool AscGraph::CheckAxisValid() const {
  int64_t id_index = 0;
  const auto axes = GetAllAxis();
  for (const auto &axis : axes) {
    CHECK_FALSE_RETURN_FALSE(axis, "Axis ptr is null, index[%ld].", id_index);
    CHECK_FALSE_RETURN_FALSE(axis->id == id_index, "Axis index[%ld] is not equal to id[%ld].", id_index, axis->id);
    id_index++;
  }
  int32_t node_index = -1;
  for (const auto &node : GetAllNodes()) {
    node_index++;
    CHECK_FALSE_RETURN_FALSE(node != nullptr, "Node ptr is null, index[%d].", node_index);
    std::set<int64_t> sched_axis_set;
    int32_t sched_axis_index = -1;
    for (const auto &sched_axis : node->attr.sched.axis) {
      sched_axis_index++;
      CHECK_FALSE_RETURN_FALSE(sched_axis >= 0L, "Invalid sched axis[%ld], node_name[%s], index[%d].", sched_axis,
                               node->GetName().c_str(), sched_axis_index);
      CHECK_FALSE_RETURN_FALSE(sched_axis < static_cast<int64_t>(axes.size()),
                               "Invalid sched axis[%ld], node_name[%s], index[%d].", sched_axis,
                               node->GetName().c_str(), sched_axis_index);
      const auto iter = sched_axis_set.find(sched_axis);
      CHECK_FALSE_RETURN_FALSE(iter == sched_axis_set.cend(), "Redundant sched axis[%ld], node_name[%s].", sched_axis,
                               node->GetName().c_str());
      sched_axis_set.insert(sched_axis);
    }
    int32_t output_index = -1;
    for (const auto &tensor : node->outputs()) {
      output_index++;
      CHECK_FALSE_RETURN_FALSE(tensor != nullptr, "Tensor ptr is null, index[%d], node name[%s].", output_index,
                               node->GetName().c_str());
      CHECK_FALSE_RETURN_FALSE(tensor->attr.axis.size() == tensor->attr.repeats.size(),
                               "Tensor axis size[%zu] is not equal to repeat size[%zu], index[%d], node name[%s].",
                               tensor->attr.axis.size(), tensor->attr.repeats.size(), output_index,
                               node->GetName().c_str());
      CHECK_FALSE_RETURN_FALSE(tensor->attr.axis.size() == tensor->attr.strides.size(),
                               "Tensor axis size[%zu] is not equal to stride size[%zu], index[%d], node name[%s].",
                               tensor->attr.axis.size(), tensor->attr.strides.size(), output_index,
                               node->GetName().c_str());
      for (const auto &axis : tensor->attr.axis) {
        CHECK_FALSE_RETURN_FALSE(axis >= 0, "Invalid tensor axis[%ld].", axis);
        CHECK_FALSE_RETURN_FALSE(axis < static_cast<int64_t>(axes.size()), "Invalid tensor axis[%ld].", axis);
      }
      for (const auto &vectorized_axis : tensor->attr.vectorized_axis) {
        CHECK_FALSE_RETURN_FALSE(vectorized_axis >= 0, "Invalid tensor vectorized_axis[%ld].", vectorized_axis);
        CHECK_FALSE_RETURN_FALSE(vectorized_axis < static_cast<int64_t>(axes.size()),
                                 "Invalid tensor vectorized_axis[%ld].", vectorized_axis);
      }
    }
  }
  return true;
}

bool AscGraph::CheckExecOrderValid() const {
  std::set<int64_t> exec_order_set;
  for (const auto &node : GetAllNodes()) {
    const auto exec_order = node->attr.sched.exec_order;
    const auto iter = exec_order_set.find(exec_order);
    CHECK_FALSE_RETURN_FALSE(iter == exec_order_set.end(), "Redundant exec_order[%ld].", exec_order);
    exec_order_set.insert(exec_order);
  }
  return true;
}

bool AscGraph::CheckTensorValid() const {
  for (const auto &node : GetAllNodes()) {
    int32_t output_index = -1;
    for (const auto &tensor : node->outputs()) {
      output_index++;
      if (tensor->attr.mem.alloc_type == AllocType::kAllocTypeGlobal) {
        continue;
      }
      if ((tensor->attr.buf.id != kIdNone) && (tensor->attr.que.id == kIdNone)) {
        continue;
      }
      if ((tensor->attr.buf.id == kIdNone) && (tensor->attr.que.id != kIdNone)) {
        CHECK_FALSE_RETURN_FALSE(tensor->attr.que.depth > 0, "Invalid que depth[%ld], tensor index[%d], node[%s].",
                                 tensor->attr.que.depth, output_index, node->GetName().c_str());
        CHECK_FALSE_RETURN_FALSE(tensor->attr.que.buf_num > 0, "Invalid que buf_num[%ld], tensor index[%d], node[%s].",
                                 tensor->attr.que.buf_num, output_index, node->GetName().c_str());
        continue;
      }
      GE_LOGE("Invalid mem, alloc type[%d], que id[%ld], buf id[%ld], tensor index[%d], node[%s].",
              static_cast<int32_t>(tensor->attr.mem.alloc_type), tensor->attr.que.id, tensor->attr.buf.id,
              output_index, node->GetName().c_str());
      return false;
    }
  }
  return true;
}

bool AscGraph::CheckNodeConnectionValid() const {
  for (const auto &node : GetAllNodes()) {
    for (uint32_t index = 0U; index < node->inputs.Size(); index++) {
      CHECK_FALSE_RETURN_FALSE(node->GetInDataAnchor(index) != nullptr, "Input is not connected, index[%u], node[%s].",
                               index, node->GetName().c_str());
      CHECK_FALSE_RETURN_FALSE(node->GetInDataAnchor(index)->GetPeerOutAnchor() != nullptr,
                               "Input is not connected, index[%u], node[%s].", index, node->GetName().c_str());
    }
  }
  return true;
}

bool AscGraph::CheckValid() const {
  if (!CheckExprValid()) {
    return false;
  }
  if (!CheckAxisValid()) {
    return false;
  }
  if (!CheckTensorValid()) {
    return false;
  }
  if (!CheckNodeConnectionValid()) {
    return false;
  }
  return true;
}

TransInfoRoadOfGraph AscGraph::GetAllAxisTransInfo() const {
  ASCIR_ASSERT_NOTNULL(impl_);
  return impl_->GetAllAxisTransInfo();
}

AscOpOutput AscGraph::CreateContiguousData(const char *name,
                                           const ge::DataType &dt,
                                           const vector<ge::Axis> &axes,
                                           const vector<std::vector<int64_t>> &axis_continuous_map,
                                           const Format &format) {
  ASCIR_ASSERT_NOTNULL(impl_);
  return impl_->CreateContiguousData(name, dt, axes, axis_continuous_map, format);
}

AscOpOutput AscGraph::CreateContiguousData(const char *name,
                                           const ge::DataType &dt,
                                           const std::vector<ge::Axis> &axes,
                                           const ge::Format &format) {
  return CreateContiguousData(name, dt, axes, kOneAxisContinuousInfo, format);
}

AscOpOutput AscGraph::CreateContiguousOut(const char *name,
                                          const ge::DataType &dt,
                                          const std::vector<ge::Axis> &axes,
                                          const ge::Format &format) {
  ASCIR_ASSERT_NOTNULL(impl_);
  return impl_->CreateContiguousOut(name, dt, axes, format);
}

void AddEdgeForNode(const ge::Operator &src_op, int32_t src_index, ge::Operator &dst_op, int32_t dst_index) {
  auto src_node = ge::NodeUtilsEx::GetNodeFromOperator(src_op);
  auto dst_node = ge::NodeUtilsEx::GetNodeFromOperator(dst_op);
  ASCIR_ASSERT_NOTNULL(src_node);
  graphStatus ret = GRAPH_SUCCESS;
  if (dst_node == nullptr) {
    auto com_graph = src_node->GetOwnerComputeGraph();
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(com_graph);
    auto dst_op_desc = ge::OpDescUtils::GetOpDescFromOperator(dst_op);
    auto dst_asc_node = std::make_shared<AscNode>(dst_op_desc, com_graph);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(dst_asc_node);
    (void)dst_asc_node->Init();
    ConstNodePtr const_dst_node = dst_asc_node;
    CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED,
      ge::NodeUtilsEx::SetNodeToOperator(dst_op, const_dst_node) == GRAPH_SUCCESS);
    dst_node = com_graph->AddNode(dst_asc_node);
    ret = ge::GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_index), dst_node->GetInDataAnchor(dst_index));
    // update tensors
    (void) dst_asc_node->inputs();
    (void) dst_asc_node->outputs();
  } else {
    ret = ge::GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_index), dst_node->GetInDataAnchor(dst_index));
  }
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, ret == GRAPH_SUCCESS);
}

int64_t AscOpOutput::GenContainerId() {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_);
  return ascir::cg::CodeGenUtils::GenNextContainerId(*op_);
}

void AscOpOutput::UseTQue(const Position pos, const int64_t depth, const int64_t buf_num, const int64_t id) {
  CHECK_BOOL_WITH_THROW_EXCEPTION(PARAM_INVALID, !HasBindToContainer(),
                                  " this tensor has been bound to a que, can not use any other que.");
  CHECK_BOOL_WITH_THROW_EXCEPTION(PARAM_INVALID, buf_num > 0, "input buf_num should be greater than 0.");
  CHECK_BOOL_WITH_THROW_EXCEPTION(PARAM_INVALID, buf_num < static_cast<int64_t>(INT32_MAX),
                                  "input buf_num should be less than INT32_MAX.");
  CHECK_BOOL_WITH_THROW_EXCEPTION(PARAM_INVALID, depth > 0, "input depth should be greater than 0.");
  CHECK_BOOL_WITH_THROW_EXCEPTION(PARAM_INVALID, depth < static_cast<int64_t>(INT32_MAX),
                                  "input depth should be less than INT32_MAX.");
  mem->position = pos;
  mem->alloc_type = AllocType::kAllocTypeQueue;
  buf->id = kIdNone;
  que->depth = depth;
  que->buf_num = buf_num;
  if (id == kIdNone) {
    que->id = GenContainerId();
  } else {
    que->id = id;
  }
}

void AscOpOutput::UseTBuf(const Position pos, const int64_t id) {
  CHECK_BOOL_WITH_THROW_EXCEPTION(PARAM_INVALID, !HasBindToContainer(),
                                  " this tensor has been bound to a buf, can not use any other buf.");
  mem->position = pos;
  mem->alloc_type = AllocType::kAllocTypeBuffer;
  que->id = kIdNone;
  if (id == kIdNone) {
    buf->id = GenContainerId();
  } else {
    buf->id = id;
  }
}

bool AscOpOutput::HasBindToContainer() const {
  bool has_bind_que = (que->id != kIdNone);
  bool has_bind_buf = (buf->id != kIdNone);
  // 1.if alloc type has set to que or buffer means has binding to a container
  // 2.if que/buf is valid, means also means has binding to a container
  return ((mem->alloc_type == AllocType::kAllocTypeQueue) || (mem->alloc_type == AllocType::kAllocTypeBuffer)) &&
      (has_bind_que || has_bind_buf);
}

void LinkByIrIndex(const ge::Operator &src_op, uint32_t src_ir_index, ge::Operator &dst_op, uint32_t dst_ir_index,
                   uint32_t dynamic_index) {
  auto dst_op_desc = ge::OpDescUtils::GetOpDescFromOperator(dst_op);
  auto src_op_desc = ge::OpDescUtils::GetOpDescFromOperator(src_op);
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, src_op_desc != nullptr);
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, dst_op_desc != nullptr);
  const std::vector<std::pair<std::string, ge::IrInputType>> &ir_inputs = dst_op_desc->GetIrInputs();
  const std::vector<std::pair<std::string, ge::IrOutputType>> &ir_outputs = src_op_desc->GetIrOutputs();
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, dst_ir_index < ir_inputs.size(),
                                  "dst_ir_index = %u, ir_inputs size = %zu", dst_ir_index, ir_inputs.size());
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, src_ir_index < ir_outputs.size(),
                                  "src_ir_index = %u, ir_outputs size = %zu", dst_ir_index, ir_outputs.size());
  auto &name_to_input_idx = dst_op_desc->MutableAllInputName();
  auto &name_to_output_idx = src_op_desc->MutableAllOutputName();
  if (ir_inputs[dst_ir_index].second == ge::IrInputType::kIrInputDynamic) {
    std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
    (void) ge::OpDescUtils::GetIrInputInstanceDescRange(dst_op_desc, ir_input_2_range);
    uint32_t dst_index = ir_input_2_range[dst_ir_index].first + dynamic_index;
    uint32_t src_index = name_to_output_idx[ir_outputs[src_ir_index].first];
    dst_op.SetInput(dst_index, src_op, src_index);
    AddEdgeForNode(src_op, static_cast<int32_t>(src_index), dst_op, static_cast<int32_t>(dst_index));
  } else {
    uint32_t dst_index = name_to_input_idx[ir_inputs[dst_ir_index].first];
    uint32_t src_index = name_to_output_idx[ir_outputs[src_ir_index].first];
    dst_op.SetInput(dst_index, src_op, src_index);
    AddEdgeForNode(src_op, static_cast<int32_t>(src_index), dst_op, static_cast<int32_t>(dst_index));
  }
}

void SetDynamicInputNumByIrIndex(ge::Operator &op, uint32_t ir_index, uint32_t dynamic_num) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  const std::vector<std::pair<std::string, ge::IrInputType>> &ir_inputs = op_desc->GetIrInputs();
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, ir_index < ir_inputs.size());
  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, ir_inputs[ir_index].second == ge::IrInputType::kIrInputDynamic);
  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  (void) ge::OpDescUtils::GetIrInputInstanceDescRange(op_desc, ir_input_2_range);

  CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED, ir_input_2_range[ir_index].second < dynamic_num,
                                  "Dynamic index [%u] is invalid.", dynamic_num);
  op_desc->AddDynamicInputDescByIndex(ir_inputs[ir_index].first, dynamic_num, ir_input_2_range[ir_index].first);
  GELOGD("Add DynamicInputDescByIndex for op_desc[%s], ir_index[%u], dynamic_num[%u]", op_desc->GetNamePtr(), ir_index,
         dynamic_num);
}

graphStatus AscGraphAttr::SerializeAttr(ascend_ir::proto::AscGraphAttrGroupsDef &asc_graph_group) {
  asc_graph_group.set_tiling_key(tiling_key);
  // axis serialize
  auto axis_defs = asc_graph_group.axis();
  for (const auto &ax : axis) {
    auto ax_def = asc_graph_group.add_axis();
    ax_def->set_id(ax->id);
    ax_def->set_name(ax->name);
    ax_def->set_axis_type(ax->type);
    ax_def->set_bind_block(ax->bind_block);
    ax_def->set_size(ax->size.ToString());
    ax_def->set_align(ax->align);
    for (const auto fm : ax->from) {
      ax_def->add_from(fm);
    }
    ax_def->set_split_pair_other_id(ax->split_pair_other_id);
    ax_def->set_allow_oversize_axis(ax->allow_oversize_axis);
    ax_def->set_allow_unaligned_tail(ax->allow_unaligned_tail);
  }
  asc_graph_group.set_type(static_cast<int64_t>(type));
  GELOGD("Graph serialization successful, tiling_key[%ld] type[%ld]", tiling_key, static_cast<int64_t>(type));
  return GRAPH_SUCCESS;
}

graphStatus AscGraphAttr::DeserializeAttr(const ascend_ir::proto::AscGraphAttrGroupsDef &asc_graph_group) {
  tiling_key = asc_graph_group.tiling_key();
  for (const auto &ax : asc_graph_group.axis()) {
    auto new_axis = std::make_shared<Axis>();
    GE_ASSERT_NOTNULL(new_axis);
    new_axis->id = ax.id();
    new_axis->name = ax.name();
    new_axis->type = static_cast<Axis::Type>(ax.axis_type());
    new_axis->bind_block = ax.bind_block();
    new_axis->size = Expression::Deserialize(ax.size().c_str());
    new_axis->align = ax.align();
    for (const auto &fm : ax.from()) {
      new_axis->from.emplace_back(fm);
    }
    new_axis->split_pair_other_id = ax.split_pair_other_id();
    new_axis->allow_oversize_axis = ax.allow_oversize_axis();
    new_axis->allow_unaligned_tail = ax.allow_unaligned_tail();
    axis.emplace_back(new_axis);
  }
  type = static_cast<ge::AscGraphType>(asc_graph_group.type());
  GELOGD("Graph deserialization successful, tiling_key[%ld], type[%ld]", tiling_key, asc_graph_group.type());
  return GRAPH_SUCCESS;
}

graphStatus AscNodeAttr::SerializeAttr(ascend_ir::proto::AscNodeAttrGroupsDef &asc_node_group) const{
  asc_node_group.set_name(name);
  asc_node_group.set_type(type);
  auto sched_def = asc_node_group.mutable_sched();
  sched_def->set_exec_order(sched.exec_order);
  for (const int64_t axis_id : sched.axis) {
    sched_def->add_axis(axis_id);
  }
  sched_def->set_loop_axis(sched.loop_axis);
  sched_def->set_exec_order(sched.exec_order);
  auto api_def = asc_node_group.mutable_api();
  api_def->set_type(static_cast<int32_t>(api.type));
  api_def->set_compute_type(static_cast<int32_t>(api.compute_type));
  api_def->set_unit(static_cast<int32_t>(api.unit));
  if (ir_attr != nullptr) {
    ir_attr->Serialize(*(asc_node_group.mutable_ir_attr_def()));
  }
  for (const auto &tmp_buffer : tmp_buffers) {
    auto tmp_buffer_def = asc_node_group.add_tmp_buffers();
    auto buf_desc_def = tmp_buffer_def->mutable_buf_desc();
    buf_desc_def->set_size(tmp_buffer.buf_desc.size.ToString());
    buf_desc_def->set_life_time_axis_id(tmp_buffer.buf_desc.life_time_axis_id);
    auto mem_def = tmp_buffer_def->mutable_mem();
    mem_def->set_tensor_id(tmp_buffer.mem.tensor_id);
    mem_def->set_alloc_type(static_cast<int64_t>(tmp_buffer.mem.alloc_type));
    mem_def->set_position(static_cast<int64_t>(tmp_buffer.mem.position));
    mem_def->set_hardware(static_cast<int64_t>(tmp_buffer.mem.hardware));
    mem_def->set_reuse_id(static_cast<int64_t>(tmp_buffer.mem.reuse_id));
    for (const int64_t buf_id : tmp_buffer.mem.buf_ids) {
      mem_def->add_buf_ids(buf_id);
    }
    mem_def->set_name(tmp_buffer.mem.name);
  }
  GELOGD("Serialize node[%s:%s] success.", name.c_str(), type.c_str());
  return GRAPH_SUCCESS;
}

graphStatus AscNodeAttr::DeserializeAttr(const ascend_ir::proto::AscNodeAttrGroupsDef &asc_node_group) {
  name = asc_node_group.name();
  type = asc_node_group.type();
  const auto &sched_def = asc_node_group.sched();
  for (const auto &ax : sched_def.axis()) {
    sched.axis.emplace_back(ax);
  }
  sched.loop_axis = sched_def.loop_axis();
  sched.exec_order = sched_def.exec_order();
  const auto &api_def = asc_node_group.api();
  api.type = static_cast<ApiType>((api_def.type()));
  api.compute_type = static_cast<ComputeType>(api_def.compute_type());
  api.unit = static_cast<ComputeUnit>(api_def.unit());
  if (asc_node_group.has_ir_attr_def()) {
    if (ir_attr == nullptr) {
      ir_attr = ComGraphMakeUnique<AscIrAttrDefBase>();
    }
    GE_ASSERT_NOTNULL(ir_attr);
    ir_attr->Deserialize(asc_node_group.ir_attr_def());
  }
  for (const auto &tmp_buffer_def : asc_node_group.tmp_buffers()) {
    TmpBufDesc new_tmp_buffer_desc;
    new_tmp_buffer_desc.size = Expression::Deserialize(tmp_buffer_def.buf_desc().size().c_str());
    new_tmp_buffer_desc.life_time_axis_id = tmp_buffer_def.buf_desc().life_time_axis_id();
    MemAttr new_mem_attr;
    new_mem_attr.name = tmp_buffer_def.mem().name();
    new_mem_attr.tensor_id = tmp_buffer_def.mem().tensor_id();
    new_mem_attr.alloc_type = static_cast<AllocType>(tmp_buffer_def.mem().alloc_type());
    new_mem_attr.position = static_cast<Position>(tmp_buffer_def.mem().position());
    new_mem_attr.hardware = static_cast<MemHardware>(tmp_buffer_def.mem().hardware());
    new_mem_attr.reuse_id = tmp_buffer_def.mem().reuse_id();
    for (const int64_t buf_id : tmp_buffer_def.mem().buf_ids()) {
      new_mem_attr.buf_ids.emplace_back(buf_id);
    }
    TmpBuffer new_tmp_buffer;
    new_tmp_buffer.buf_desc = new_tmp_buffer_desc;
    new_tmp_buffer.mem = new_mem_attr;
    tmp_buffers.emplace_back(new_tmp_buffer);
  }
  return GRAPH_SUCCESS;
}

graphStatus AscTensorAttr::SerializeAttr(ascend_ir::proto::AscTensorAttrGroupsDef &asc_tensor_group) {
  asc_tensor_group.set_dtype(static_cast<int64_t>(dtype));
  for (const int64_t axis_id : axis) {
    asc_tensor_group.add_axis_ids(axis_id);
  }
  for (const auto &repeat : repeats) {
    asc_tensor_group.add_repeats(repeat.ToString());
  }
  for (const auto &stride : strides) {
    asc_tensor_group.add_strides(stride.ToString());
  }
  for (const auto &vectorized_axis_id : vectorized_axis) {
    asc_tensor_group.add_vectorized_axis(vectorized_axis_id);
  }
  for (const auto &vectorized_stride : vectorized_strides) {
    asc_tensor_group.add_vectorized_strides(vectorized_stride.ToString());
  }
  auto mem_def = asc_tensor_group.mutable_mem();
  mem_def->set_tensor_id(mem.tensor_id);
  mem_def->set_alloc_type(static_cast<int64_t>(mem.alloc_type));
  mem_def->set_position(static_cast<int64_t>(mem.position));
  mem_def->set_hardware(static_cast<int64_t>(mem.hardware));
  for (const int64_t buf_id : mem.buf_ids) {
    mem_def->add_buf_ids(buf_id);
  }
  mem_def->set_name(mem.name);
  auto que_def = asc_tensor_group.mutable_que();
  que_def->set_id(que.id);
  que_def->set_depth(que.depth);
  que_def->set_buf_num(que.buf_num);
  que_def->set_name(que.name);
  auto buf_def = asc_tensor_group.mutable_buf();
  buf_def->set_id(buf.id);
  buf_def->set_name(buf.name);
  auto opt_def = asc_tensor_group.mutable_opt();
  opt_def->set_reuse_id(opt.reuse_id);
  opt_def->set_ref_tensor(opt.ref_tensor);
  opt_def->set_merge_scope(opt.merge_scope);
  return GRAPH_SUCCESS;
}

graphStatus AscTensorAttr::DeserializeAttr(const ascend_ir::proto::AscTensorAttrGroupsDef &asc_tensor_group,
                                           GeTensorDesc *tensor_desc) {
  dtype.tensor_desc_ = tensor_desc;
  if (dtype.tensor_desc_ != nullptr) {
    dtype.tensor_desc_->SetDataType(static_cast<ge::DataType>(asc_tensor_group.dtype()));
  }
  for (const auto &axis_id : asc_tensor_group.axis_ids()) {
    axis.emplace_back(axis_id);
  }
  const auto &repeat_defs = asc_tensor_group.repeats();
  for (const auto &repeat : repeat_defs) {
    repeats.emplace_back(Expression::Deserialize(repeat.c_str()));
  }
  const auto &strides_defs = asc_tensor_group.strides();
  for (const auto &stride : strides_defs) {
    strides.emplace_back(Expression::Deserialize(stride.c_str()));
  }
  const auto &vectorized_axis_ids = asc_tensor_group.vectorized_axis();
  for (const auto &vectorized_axis_id : vectorized_axis_ids) {
    vectorized_axis.emplace_back(vectorized_axis_id);
  }
  const auto &vectorized_strides_def = asc_tensor_group.vectorized_strides();
  for (const auto &vectorized_stride : vectorized_strides_def) {
    vectorized_strides.emplace_back(Expression::Deserialize(vectorized_stride.c_str()));
  }
  const auto &mem_def = asc_tensor_group.mem();
  mem.name = mem_def.name();
  mem.tensor_id = mem_def.tensor_id();
  mem.alloc_type = static_cast<AllocType>(mem_def.alloc_type());
  mem.position = static_cast<Position>(mem_def.position());
  mem.hardware = static_cast<MemHardware>(mem_def.hardware());
  for (const int64_t buf_id : mem_def.buf_ids()) {
    mem.buf_ids.emplace_back(buf_id);
  }
  mem.name = mem_def.name();
  const auto &que_def = asc_tensor_group.que();
  que.id = que_def.id();
  que.name = que_def.name();
  que.depth = que_def.depth();
  que.buf_num = que_def.buf_num();
  const auto &buf_def = asc_tensor_group.buf();
  buf.id = buf_def.id();
  buf.name = buf_def.name();
  const auto &opt_def = asc_tensor_group.opt();
  opt.merge_scope = opt_def.merge_scope();
  opt.ref_tensor = opt_def.ref_tensor();
  opt.reuse_id = opt_def.reuse_id();
  return GRAPH_SUCCESS;
}

graphStatus AscIrAttrDefBase::Serialize(ascend_ir::proto::AscIrAttrDef &asc_ir_attr_def) {
  std::map<std::string, AnyValue> names_to_attr;
  attr_store_.GetAllAttrs(names_to_attr);
  auto &attr_map = *asc_ir_attr_def.mutable_attr();
  for (const auto &pair:names_to_attr) {
    const auto serializer = AttrSerializerRegistry::GetInstance().GetSerializer(
        pair.second.GetValueTypeId());
    GE_ASSERT_NOTNULL(serializer);
    proto::AttrDef attr_def;
    GE_ASSERT_GRAPH_SUCCESS(serializer->Serialize(pair.second, attr_def));
    attr_map[pair.first] = attr_def;
  }
  return GRAPH_SUCCESS;
}

graphStatus AscIrAttrDefBase::Deserialize(const ascend_ir::proto::AscIrAttrDef &asc_ir_attr_def) {
  const auto &attr_map = asc_ir_attr_def.attr();
  for (const auto &pair:attr_map) {
    const auto deserializer = AttrSerializerRegistry::GetInstance()
        .GetDeserializer(pair.second.value_case());
    GE_ASSERT_NOTNULL(deserializer);
    auto attr_value = attr_store_.GetOrCreateAnyValue(pair.first);
    GE_ASSERT_NOTNULL(attr_value);
    GE_ASSERT_GRAPH_SUCCESS(deserializer->Deserialize(pair.second, *attr_value));
  }
  return GRAPH_SUCCESS;
}

std::unique_ptr<AscIrAttrDefBase> AscIrAttrDefBase::Clone() {
  auto ptr = ComGraphMakeUnique<AscIrAttrDefBase>();
  ASCIR_ASSERT_NOTNULL(ptr);
  ptr->attr_store_ = this->attr_store_;
  return ptr;
}

graphStatus AscDataIrAttrDef::GetIndex(int64_t &index) const {
  auto value = attr_store_.GetAnyValue(kDataIndex);
  GE_WARN_ASSERT(value != nullptr);
  return value->GetValue(index);
}

graphStatus AscDataIrAttrDef::SetIndex(int64_t index) {
  auto value = attr_store_.GetOrCreateAnyValue(kDataIndex);
  ASCIR_ASSERT_NOTNULL(value);
  return value->SetValue(index);
}
}  // namespace ge
