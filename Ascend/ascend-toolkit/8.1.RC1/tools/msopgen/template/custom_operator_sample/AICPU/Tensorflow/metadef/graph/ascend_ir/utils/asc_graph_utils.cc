/* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "graph/ascend_ir/utils/asc_graph_utils.h"
#include <google/protobuf/text_format.h>
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/ascend_ir/core/ascend_ir_impl.h"
#include "proto/ascend_ir.pb.h"

namespace ge {
namespace {
graphStatus EstablishAscNodeAndEdges(const ascend_ir::proto::AscGraphDef &asc_graph_def, AscGraph &out_asc_graph) {
  auto &asc_nodes = asc_graph_def.asc_node();
  // 1. Add AscNodes to AscGraph
  for (const auto &asc_node : asc_nodes) {
    auto &node_attr = asc_node.attr();
    auto &ir_attr = asc_node.ir_def();
    auto &op_name = node_attr.name();
    auto &op_type = ir_attr.type();
    OpDescBuilder op_builder(op_name, op_type);
    const auto &inputs_nums = ir_attr.input_nums();
    const auto &inputs_names = ir_attr.input_names();
    const auto &input_types = ir_attr.input_ir_type();
    GE_ASSERT_TRUE(inputs_nums.size() == input_types.size(),
                   "[Build][Op] for %s failed, inputs_nums[%zu] "
                   "input_types[%zu] not equal.",
                   op_name.c_str(), inputs_nums.size(), input_types.size());
    GE_ASSERT_TRUE(inputs_nums.size() == inputs_names.size(),
                   "[Build][Op] for %s failed, inputs_nums[%zu] "
                   "inputs_names[%zu] not equal.",
                   op_name.c_str(), inputs_nums.size(), inputs_names.size());
    for (int32_t ir_id = 0; ir_id < inputs_nums.size(); ir_id++) {
      if (input_types[ir_id] == IrInputType::kIrInputDynamic) {
        op_builder.AddDynamicInput(ir_attr.input_names(ir_id), inputs_nums[ir_id]);
        GELOGD("Add dynamic input[%s] of node[%s] success, input num[%ld].", inputs_names[ir_id].c_str(),
               op_name.c_str(), inputs_nums[ir_id]);
      } else {
        op_builder.AddInput(inputs_names[ir_id]);
        GELOGD("Add input[%s] of node[%s] success.", inputs_names[ir_id].c_str(), op_name.c_str());
      }
    }
    for (auto &ir_output_name : ir_attr.output_names()) {
      op_builder.AddOutput(ir_output_name);
    }
    auto op_desc = op_builder.Build();
    GE_ASSERT_NOTNULL(op_desc, "[Build][Op] for %s failed.", op_name.c_str());
    auto op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    out_asc_graph.AddNode(op);
    GELOGD("Add node[%s:%s] inputs size[%d] output size[%d] success.", op_name.c_str(), op_type.c_str(),
           ir_attr.input_names().size(), ir_attr.output_names().size());
  }
  const auto &compute_graph = AscGraphUtils::GetComputeGraph(out_asc_graph);
  // 2. Add edge between all AscNodes
  for (const auto &asc_node : asc_nodes) {
    auto &node_attr = asc_node.attr();
    auto &dst_node_name = node_attr.name();
    int32_t dst_in_index = 0;
    for (const auto &input : asc_node.input_src()) {
      const auto &src_node_name = input.src_node_name();
      const auto &src_out_index = input.src_out_index();
      if (src_node_name.empty()) {
        GELOGW("[Get][SrcNodeName] failed of node [%s:%d]", node_attr.name().c_str(), dst_in_index);
        continue;
      }
      const auto &src_node = compute_graph->FindNode(src_node_name);
      GE_ASSERT_NOTNULL(src_node, "[Find][SrcNode] %s failed, dst_node[%s].", src_node_name.c_str(),
                        node_attr.name().c_str());
      const auto &dst_node = compute_graph->FindNode(dst_node_name);
      GE_ASSERT_NOTNULL(dst_node, "[Find][DstNode] %s failed, dst_node[%s].", src_node_name.c_str(),
                        node_attr.name().c_str());
      GE_ASSERT_GRAPH_SUCCESS(
          GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_out_index), dst_node->GetInDataAnchor(dst_in_index)),
          "[Add][Edge] failed to add edge from node[%s:%d] to node[%s:%d] failed.Possible  duplicate link.",
          src_node_name.c_str(), src_out_index, dst_node_name.c_str(), dst_in_index);
      GELOGD("[Add][Edge] from node[%s:%d] to node[%s:%d].", src_node_name.c_str(), src_out_index,
             dst_node_name.c_str(), dst_in_index);
      dst_in_index++;
    }
  }
  GELOGD("Deserialize graph[%s] success, graph node size[%zu]", out_asc_graph.GetName().c_str(),
         compute_graph->GetDirectNodesSize());
  return GRAPH_SUCCESS;
}
}
ComputeGraphPtr AscGraphUtils::GetComputeGraph(const AscGraph &asc_graph) {
  return asc_graph.impl_->GetComputeGraph();
}

graphStatus AscGraphUtils::SerializeToProto(const ge::AscGraph &asc_graph,
                                            ascend_ir::proto::AscGraphDef &asc_graph_def) {
  const auto &ge_graph = AscGraphUtils::GetComputeGraph(asc_graph);
  GE_ASSERT_NOTNULL(ge_graph);
  asc_graph_def.set_graph_name(asc_graph.GetName());
  // serialize asc graph attr
  auto asc_graph_attr_def = asc_graph_def.mutable_asc_graph_attr();
  GE_ASSERT_NOTNULL(asc_graph_attr_def);
  auto asc_graph_attr = ge_graph->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(asc_graph_attr, "[GetOrCreate][AscGraphAttr] failed, graph[%s].", asc_graph.GetName().c_str());
  asc_graph_attr->SerializeAttr(*asc_graph_attr_def);
  // serialize asc nodes
  for (const auto &node : asc_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node, "[Get][Node] failed, graph[%s]", asc_graph.GetName().c_str());
    // serialize asc node def
    auto node_def = asc_graph_def.add_asc_node();
    GE_ASSERT_NOTNULL(node_def, "[Add][AscNode] proto failed, graph[%s]", asc_graph.GetName().c_str());
    // serialize asc node attr
    const auto attr_def = node_def->mutable_attr();
    GE_ASSERT_NOTNULL(attr_def, "[Get][Attr] failed, graph[%s]", asc_graph.GetName().c_str());
    GE_ASSERT_GRAPH_SUCCESS(AscNodeSerializeUtils::SerializeAttrGroupsDef(*node, *attr_def),
                            "[Serialize][Attr] failed, graph[%s]", asc_graph.GetName().c_str());
    // serialize asc node ir def
    auto ir_def = node_def->mutable_ir_def();
    GE_ASSERT_NOTNULL(ir_def, "[Get][IrAttr] failed, graph[%s]", asc_graph.GetName().c_str());
    GE_ASSERT_GRAPH_SUCCESS(AscNodeSerializeUtils::SerializeIrDef(*node, *ir_def),
                            "[Serialize][IrAttr] failed, graph[%s]", asc_graph.GetName().c_str());
    GELOGD("[AscGraphUtils]Serialize ir attr node[%s:%s] success.", node->GetNamePtr(), ir_def->type().c_str());
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    // serialize asc in tensor
    const auto &ir_index_map = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(op_desc);
    size_t ir_id = 0UL;
    size_t cur_input_id = 0UL;
    for (const auto &ir_input : op_desc->GetIrInputs()) {
      size_t start = cur_input_id;
      size_t end = cur_input_id;
      if (ir_input.second == IrInputType::kIrInputDynamic) {
        const auto &range_iter = ir_index_map.find(ir_id);
        if (range_iter != ir_index_map.cend()) {
          start = range_iter->second.first;
          end = range_iter->second.second;
        }
      } else {
        end = start + 1;
      }
      GE_ASSERT_TRUE(start <= end);
      GE_ASSERT_TRUE(end <= op_desc->GetAllInputsSize());
      int64_t input_nums = static_cast<int64_t>(end - start);
      ir_def->add_input_nums(input_nums);
      GELOGD("Add input nums[%d, end:%zu, start:%zu] for node[%s:%s, %d]", input_nums, end, start,
             op_desc->GetNamePtr(), op_desc->GetTypePtr(), ir_id);
      for (auto id = static_cast<int32_t>(start); id < static_cast<int32_t>(end); id++) {
        const auto input_src = node_def->add_input_src();
        const auto &in_anchor = node->GetInDataAnchor(id);
        GE_ASSERT_NOTNULL(in_anchor, "[Get][InDataAnchor] failed, graph[%s]", asc_graph.GetName().c_str());
        const auto &peer_out = in_anchor->GetPeerOutAnchor();
        if (peer_out == nullptr) {
          GELOGW("[Get][PeerOut] failed of node [%s:%d]", node->GetNamePtr(), in_anchor->GetIdx());
          continue;
        }
        const auto &src_node = peer_out->GetOwnerNodeBarePtr();
        GE_ASSERT_NOTNULL(src_node, "[Get][SrcNode] failed, graph[%s]", asc_graph.GetName().c_str());
        input_src->set_src_node_name(src_node->GetName());
        input_src->set_src_out_index(peer_out->GetIdx());
        GELOGD("Set src node name[%s:%d] for node[%s:%s, %d]", src_node->GetName().c_str(), peer_out->GetIdx(),
               op_desc->GetNamePtr(), op_desc->GetTypePtr(), id);
      }
      ir_id++;
      cur_input_id = end;
    }
    // serialize asc out tensor
    for (const auto &tensor : op_desc->GetAllOutputsDescPtr()) {
      const auto tensor_attr = tensor->GetOrCreateAttrsGroup<AscTensorAttr>();
      GE_ASSERT_NOTNULL(tensor_attr, "[GetOrCreate][AscTensorAttr] failed, graph[%s]", asc_graph.GetName().c_str());
      const auto tensor_def = node_def->add_outputs();
      const auto tensor_attr_def = tensor_def->mutable_attr();
      GE_ASSERT_NOTNULL(tensor_attr_def, "[Get][TensorAttr] failed, graph[%s]", asc_graph.GetName().c_str());
      tensor_attr->SerializeAttr(*tensor_attr_def);
    }
    GELOGD("Serialize node[%s:%s] success, ir_id[%zu].", node->GetNamePtr(), node->GetTypePtr(), ir_id);
  }
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::SerializeToBinary(const AscGraph &asc_graph, std::string &output) {
  ascend_ir::proto::AscGraphDef asc_graph_def;
  GE_ASSERT_GRAPH_SUCCESS(SerializeToProto(asc_graph, asc_graph_def), "SerializeToProto failed.");
  GE_ASSERT_TRUE(asc_graph_def.SerializeToString(&output));
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::DeserializeFromBinary(const std::string &to_be_deserialized, AscGraph &out_asc_graph) {
  ascend_ir::proto::AscGraphDef asc_graph_def;
  GE_ASSERT_TRUE(asc_graph_def.ParseFromString(to_be_deserialized));
  GE_ASSERT_GRAPH_SUCCESS(DeserializeFromProto(asc_graph_def, out_asc_graph));
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::SerializeToReadable(const AscGraph &asc_graph, std::string &output) {
  ascend_ir::proto::AscGraphDef asc_graph_def;
  GE_ASSERT_GRAPH_SUCCESS(SerializeToProto(asc_graph, asc_graph_def), "SerializeToProto failed.");
  GE_ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(asc_graph_def, &output), "SerializeToReadable failed.");
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::DeserializeFromReadable(const std::string &to_be_deserialized, AscGraph &out_asc_graph) {
  ascend_ir::proto::AscGraphDef asc_graph_def;
  GE_ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(to_be_deserialized, &asc_graph_def));
  GE_ASSERT_GRAPH_SUCCESS(DeserializeFromProto(asc_graph_def, out_asc_graph));
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::DeserializeFromProto(const ascend_ir::proto::AscGraphDef &asc_graph_def,
                                                AscGraph &asc_graph) {
  auto &graph_name = asc_graph_def.graph_name();
  // 1. Add AscGraph
  asc_graph.impl_->compute_graph_->SetName(graph_name);
  GE_ASSERT_GRAPH_SUCCESS(EstablishAscNodeAndEdges(asc_graph_def, asc_graph),
                          "EstablishAscNodeAndEdges of graph[%s] failed.", graph_name.c_str());
  // deserialize graph attr
  const auto &ge_graph = GetComputeGraph(asc_graph);
  const auto asc_graph_attr = ge_graph->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(asc_graph_attr, "[GetOrCreate][AscGraphAttr] failed, graph[%s].", graph_name.c_str());
  GE_ASSERT_GRAPH_SUCCESS(asc_graph_attr->DeserializeAttr(asc_graph_def.asc_graph_attr()));
  // deserialize node attr, protobuf make sure the order as serializing
  const auto &asc_nodes = asc_graph_def.asc_node();
  int64_t node_index = 0L;
  for (const auto &asc_node : asc_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(asc_node, "[Get][Node] failed, graph[%s]", graph_name.c_str());
    // update asc node AscTensors
    asc_node->outputs();
    asc_node->inputs();
    GE_ASSERT_TRUE(node_index < asc_nodes.size(),
                   "[Deserialize][Node] failed, node_index[%ld] should less than nodes size[%zu].", node_index,
                   asc_nodes.size());
    const auto &asc_node_def = asc_nodes[static_cast<int32_t>(node_index)];
    const auto op_desc = asc_node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc, "[Get][OpDesc] failed, graph[%s]", graph_name.c_str());
    GE_ASSERT_GRAPH_SUCCESS(AscNodeDeserializeUtils::DeserializeAttrGroupsDef(asc_node_def.attr(), *asc_node));
    GE_ASSERT_GRAPH_SUCCESS(AscNodeDeserializeUtils::DeserializeIrDef(asc_node_def.ir_def(), *asc_node));
    // deserialize output tensor attr
    int32_t index = 0;
    for (const auto &output_def : asc_node_def.outputs()) {
      const auto &output_attr = output_def.attr();
      const auto output_desc = op_desc->MutableOutputDesc(index);
      GE_ASSERT_NOTNULL(output_desc, "[Get][OutputDesc] failed of node[%s:%s, %d].", op_desc->GetName().c_str(),
                        op_desc->GetType().c_str(), index);
      const auto output_tensor_attr = output_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      GE_ASSERT_NOTNULL(output_tensor_attr, "[GetOrCreate][OutputTensorAttr] failed of node[%s:%s, %d].",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str(), index);
      output_tensor_attr->DeserializeAttr(output_attr, output_desc.get());
      index++;
    }
    // deserialize input tensor attr
    index = 0;
    for (const auto &input_def : asc_node_def.input_src()) {
      const auto &src_node_name = input_def.src_node_name();
      const auto &src_node = asc_graph.FindNode(src_node_name.c_str());
      if (src_node == nullptr) {
        GELOGW("[Get][SrcNodeName] %s failed of node [%s:%d]", src_node_name.c_str(), asc_node->GetNamePtr(), index);
        continue;
      }
      const auto src_out_index = input_def.src_out_index();
      const auto &src_op = src_node->GetOpDesc();
      GE_ASSERT_NOTNULL(src_op);
      const auto &out_tensor_desc = src_op->MutableOutputDesc(src_out_index);
      const auto &out_tensor_attr = out_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      GE_ASSERT_NOTNULL(out_tensor_attr, "[GetOrCreate][InputTensorAttr] failed of node[%s:%s, %d].",
                        src_op->GetName().c_str(), src_op->GetType().c_str(), src_out_index);
      const auto input_desc = op_desc->MutableInputDesc(index);
      GE_ASSERT_NOTNULL(input_desc, "[Get][InputDesc] failed of node[%s:%s, %d].", op_desc->GetName().c_str(),
                        op_desc->GetType().c_str(), index);
      const auto input_tensor_attr = input_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
      GE_ASSERT_NOTNULL(input_tensor_attr, "[GetOrCreate][InputTensorAttr] failed of node[%s:%s, %d].",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str(), index);
      input_tensor_attr->dtype.tensor_desc_ = input_desc.get();
      *input_tensor_attr = *out_tensor_attr;
      index++;
    }
    node_index++;
  }
  return GRAPH_SUCCESS;
}
AscGraph AscGraphUtils::ConvertComputeGraphToAscGraph(const ComputeGraphPtr &compute_graph) {
  CHECK_NOTNULL_WITH_THROW_EXCEPTION(compute_graph);
  // 1. 共享名字
  AscGraph asc_graph(compute_graph->GetName().c_str());
  // 2. 转换Node到AscNode, 共享OpDesc
  std::unordered_map<std::string, NodePtr> all_new_nodes;
  for (const auto &node : compute_graph->GetDirectNode()) {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(node);
    const auto &op_desc = node->GetOpDesc();
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_desc);
    auto op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    auto dst_new_node = asc_graph.AddNode(op);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(dst_new_node);
    all_new_nodes[dst_new_node->GetName()] = dst_new_node;
  }
  // 3. 转换连边关系
  for (const auto &src_node : compute_graph->GetDirectNode()) {
    CHECK_BOOL_WITH_THROW_EXCEPTION(ge::GRAPH_FAILED,
                                    GraphUtils::RelinkGraphEdges(src_node, "", all_new_nodes) == GRAPH_SUCCESS);
  }
  // 4. 转换graph上的属性组属性
  AscGraphImpl::DoCopyAscGraphAttrImpl(compute_graph, AscGraphUtils::GetComputeGraph(asc_graph));
  return asc_graph;
}

graphStatus AscNodeSerializeUtils::SerializeIrDef(const AscNode &node, ascend_ir::proto::IrDef &ir_def) {
  ir_def.set_type(node.GetType());
  const auto &op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  for (const auto &ir_input : op_desc->GetIrInputs()) {
    ir_def.add_input_names(ir_input.first);
    ir_def.add_input_ir_type(ir_input.second);
  }
  for (const auto &ir_output : op_desc->GetIrOutputs()) {
    ir_def.add_output_names(ir_output.first);
    ir_def.add_output_ir_type(ir_output.second);
  }
  GELOGD("Serialize ir def node[%s:%s] success.", node.GetNamePtr(), ir_def.type().c_str());
  return GRAPH_SUCCESS;
}

graphStatus AscNodeSerializeUtils::SerializeAttrGroupsDef(const AscNode &node,
                                                          ascend_ir::proto::AscNodeAttrGroupsDef &asc_node_attr_groups_def) {
  const auto op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto asc_node_attr = op_desc->GetOrCreateAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(asc_node_attr);
  return asc_node_attr->SerializeAttr(asc_node_attr_groups_def);
}

graphStatus AscNodeDeserializeUtils::DeserializeIrDef(const ascend_ir::proto::IrDef &ir_def, AscNode &node) {
  const auto &type = ir_def.type();
  const auto &op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  op_desc->SetType(type);
  GE_ASSERT_EQ(ir_def.input_names_size(), ir_def.input_ir_type_size());
  for (int64_t index = 0; index < ir_def.input_names_size(); ++index) {
    op_desc->AppendIrInput(ir_def.input_names(index), static_cast<IrInputType>(ir_def.input_ir_type(index)));
  }
  GE_ASSERT_EQ(ir_def.output_names_size(), ir_def.output_ir_type_size());
  for (int64_t index = 0; index < ir_def.output_names_size(); ++index) {
    op_desc->AppendIrOutput(ir_def.output_names(index), static_cast<IrOutputType>(ir_def.output_ir_type(index)));
  }
  GELOGD("Deserialize ir def node[%s:%s] success.", node.GetNamePtr(), ir_def.type().c_str());
  return GRAPH_SUCCESS;
}
graphStatus AscNodeDeserializeUtils::DeserializeAttrGroupsDef(const ascend_ir::proto::AscNodeAttrGroupsDef &asc_node_attr_groups_def,
                                                              AscNode &node) {
  const auto op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto asc_node_attr = op_desc->GetOrCreateAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(asc_node_attr);
  GE_ASSERT_GRAPH_SUCCESS(asc_node_attr->DeserializeAttr(asc_node_attr_groups_def),
                          "[Deserialize][Attr] failed of node[%s:%s].", op_desc->GetName().c_str(),
                          op_desc->GetType().c_str());
  return GRAPH_SUCCESS;
}
graphStatus ExpressionSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  Expression expression;
  GE_ASSERT_GRAPH_SUCCESS(av.GetValue(expression));
  def.set_expression(expression.Serialize().get());
  return GRAPH_SUCCESS;
}

graphStatus ExpressionSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  return av.SetValue(Expression::Deserialize(def.expression().c_str()));
}

REG_GEIR_SERIALIZER(expression_serializer,
                    ExpressionSerializer,
                    GetTypeId<ge::Expression>(),
                    proto::AttrDef::kExpression);
}  // namespace ge
