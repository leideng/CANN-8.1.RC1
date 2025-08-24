/* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/
#include "ascend_graph_code_dumper.h"

namespace ge {
namespace ascir {
namespace {
// c++中的node name可能有/等python中无法作为对象名的非法字符，所以用这个类来根据type生成全局唯一的名字
class NameGenerator {
 public:
  static std::string GenerateUniqueName(const Node &node) {
    const std::string &type = node.GetType();
    const std::string &original_name = node.GetName();
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string &unique_name = type + "_" + std::to_string(type_counter_[type]++);
    name_mapping_[original_name] = unique_name;
    return unique_name;
  }

  static const std::unordered_map<std::string, std::string> &GetNameMapping() {
    return name_mapping_;
  }

 private:
  static std::unordered_map<std::string, int64_t> type_counter_;
  // 记录原始名字和python中的名字的映射关系
  static std::unordered_map<std::string, std::string> name_mapping_;
  static std::mutex mutex_;
};

std::unordered_map<std::string, int64_t> NameGenerator::type_counter_;
std::unordered_map<std::string, std::string> NameGenerator::name_mapping_;
std::mutex NameGenerator::mutex_;
std::string GetOutputName(const NodePtr &src_node, uint32_t idx) {
  const auto &idx2name = src_node->GetOpDesc()->GetAllOutputIndexToName();
  auto out_name_iter = idx2name.find(idx);
  GE_ASSERT_TRUE(out_name_iter != idx2name.end());
  return out_name_iter->second;
}

std::string GetPythonNodeNameByOriginName(const std::string &origin_name) {
  const auto &name_mapping_info = NameGenerator::GetNameMapping();
  const auto &iter = name_mapping_info.find(origin_name);
  if (iter == name_mapping_info.end()) {
    GELOGW("%s has not been added to name map, may be topo is wrong");
    return "";
  }
  return iter->second;
}

void GenerateInputCode(const std::string &op_name,
                       const std::string &input_name,
                       const NodePtr &src_node,
                       uint32_t out_idx,
                       std::ofstream &output_file) {
  std::string out_name = GetOutputName(src_node, out_idx);
  output_file << op_name << "." << input_name << " = " << GetPythonNodeNameByOriginName(src_node->GetName()) << "."
              << out_name << "\n";
}

Status GenerateDynamicInputCode(const Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>> &src_nodes,
                                size_t start_index, size_t count,
                                const std::string &op_name,
                                const std::string &input_name,
                                std::ofstream &output_file) {
  std::string dynamic_inputs_code = "[";
  for (size_t i = start_index; i < start_index + count; ++i) {
    GE_ASSERT_TRUE(i < src_nodes.size());
    const auto &src_node = src_nodes.at(i).first;
    uint32_t out_idx = src_nodes.at(i).second->GetIdx();
    std::string out_name = GetOutputName(src_node, out_idx);
    dynamic_inputs_code += GetPythonNodeNameByOriginName(src_node->GetName()) + "." + out_name;
    if (i < start_index + count - 1) {
      dynamic_inputs_code += ", ";
    }
  }
  dynamic_inputs_code += "]";
  output_file << op_name << "." << input_name << " = " << dynamic_inputs_code << "\n";
  return SUCCESS;
}

std::string GenerateDataTypeCode(ge::DataType dtype) {
  static const std::map<ge::DataType, std::string> ge_dtype_2_python_type = {
      {ge::DT_FLOAT16, "ascir.dtypes.float16"},
      {ge::DT_FLOAT, "ascir.dtypes.float32"},
      {ge::DT_INT64, "ascir.dtypes.int64"},
      {ge::DT_INT8, "ascir.dtypes.int8"},
      {ge::DT_INT32, "ascir.dtypes.int32"},
  };
  auto iter = ge_dtype_2_python_type.find(dtype);
  GE_WARN_ASSERT(iter != ge_dtype_2_python_type.end(),
                 "DataType [%s] is not supported by python now", TypeUtils::DataTypeToSerialString(dtype).c_str());
  return iter->second;
}

std::string GenerateAxisCode(const std::vector<int64_t> &axis,
                             const std::vector<std::pair<std::string, std::string>> &axis_infos) {
  std::string axis_code = "[";
  for (size_t i = 0; i < axis.size(); ++i) {
    GE_ASSERT_TRUE(axis[i] >= 0);
    GE_ASSERT_TRUE(static_cast<size_t>(axis[i]) < axis_infos.size());
    axis_code += axis_infos[axis[i]].first;
    if (i < axis.size() - 1) {
      axis_code += ", ";
    }
  }
  axis_code += "]";
  return axis_code;
}

std::string GenerateAxisSizeCode(const std::vector<int64_t> &axis, const std::vector<Expression> &repeats,
                                 const std::vector<std::pair<std::string, std::string>> &axis_infos) {
  GE_WARN_ASSERT(axis.size() == repeats.size(),
                 "Axis size %zu should be equal with repeat size %zu",
                 axis.size(), repeats.size());
  std::string axis_size_code = "[";
  for (size_t i = 0; i < axis.size(); ++i) {
    if (repeats[i].IsConstExpr()) {
      axis_size_code += repeats[i].Str().get();
    } else {
      GE_ASSERT_TRUE(axis[i] >= 0);
      GE_ASSERT_TRUE(static_cast<size_t>(axis[i]) < axis_infos.size());
      axis_size_code += axis_infos[axis[i]].second;
    }
    if (i < axis.size() - 1) {
      axis_size_code += ", ";
    }
  }
  axis_size_code += "]";
  return axis_size_code;
}

std::string GenerateAxisStrideCode(const std::vector<ge::Expression> &strides) {
  std::string axis_strides_code = "[";
  for (size_t i = 0; i < strides.size(); ++i) {
    axis_strides_code += strides[i].Str().get();
    if (i < strides.size() - 1) {
      axis_strides_code += ", ";
    }
  }
  axis_strides_code += "]";
  return axis_strides_code;
}
}

void PythonCodeDumper::GenerateHeader(std::ofstream &output_file) {
  output_file << "# Python code to construct AscGraph\n";
  output_file << "from pyautofuse import ascir\n";
  output_file << "from pyautofuse import Autofuser, AutofuserOptions\n\n";
}

Status PythonCodeDumper::GenerateNodeCode(const NodePtr &node, std::ofstream &output_file) {
  GE_ASSERT_NOTNULL(node);
  GELOGD("Start to gen node code for %s %s", node->GetNamePtr(), node->GetTypePtr());
  node_name_of_python_ = NameGenerator::GenerateUniqueName(*node);
  if (node->GetInDataNodesSize() == 0U) {
    output_file << node_name_of_python_ << " = ascir.ops." << node->GetType() << "(" << "\"" << node->GetName() << "\""
                << ", graph)" << std::endl;
  } else {
    // 有数据输入的节点，不需要graph的入参，通过连边时加入graph中
    output_file << node_name_of_python_ << " = ascir.ops." << node->GetType() << "(" << "\"" << node->GetName() << "\""
                << ")" << std::endl;
  }
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto &&node_attr_group = op_desc->GetOrCreateAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(node_attr_group);
  if (!node_attr_group->sched.axis.empty()) {
    std::string axis_code;
    axis_code.push_back('[');
    for (size_t i = 0U; i < node_attr_group->sched.axis.size(); ++i) {
      auto one_axis = node_attr_group->sched.axis[i];
      GE_ASSERT_TRUE(one_axis >= 0);
      GE_ASSERT_TRUE(static_cast<size_t>(one_axis) < axis_infos_.size());
      axis_code += axis_infos_[one_axis].first;
      if (i < node_attr_group->sched.axis.size() - 1) {
        axis_code += ", ";
      }
    }
    axis_code.push_back(']');
    output_file << node_name_of_python_ << ".attr.sched.axis = " << axis_code << std::endl;
  }
  return SUCCESS;
}

Status PythonCodeDumper::GenerateDataEdgeCode(const Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>> &src_nodes,
                                              const NodePtr &dst_node, std::ofstream &output_file) {
  const auto &op_desc = dst_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  if (src_nodes.empty()) {
    GELOGD("[%s:%s] has no input.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return SUCCESS;
  }
  GELOGD("Start to add input for node [%s:%s]", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  const auto &ir_inputs = op_desc->GetIrInputs();
  size_t ir_input_index = 0U;
  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  GE_ASSERT_GRAPH_SUCCESS(ge::OpDescUtils::GetIrInputRawDescRange(op_desc, ir_input_2_range));

  for (size_t index = 0; index < src_nodes.size(); ++ir_input_index) {
    const auto &ir_input_2_range_iter = ir_input_2_range.find(ir_input_index);
    GE_ASSERT_TRUE(ir_input_2_range_iter != ir_input_2_range.end());
    GELOGI("ir input:%zu with range [%zu, %zu)", ir_input_index, ir_input_2_range_iter->second.first,
           ir_input_2_range_iter->second.first + ir_input_2_range_iter->second.second);
    GE_ASSERT_TRUE(ir_input_index < ir_inputs.size());
    const auto &ir_input_name_2_input_type = ir_inputs[ir_input_index];
    const auto &ir_input_type = ir_input_name_2_input_type.second;
    const auto &input_name = ir_input_name_2_input_type.first;
    if (ir_input_type == ge::IrInputType::kIrInputRequired) {
      GE_ASSERT_EQ(ir_input_2_range_iter->second.second, 1U);
      const auto &src_node = src_nodes.at(index).first;
      uint32_t out_idx = src_nodes.at(index).second->GetIdx();
      GenerateInputCode(node_name_of_python_, input_name, src_node, out_idx, output_file);
      ++index;
    } else if (ir_input_type == ge::IrInputType::kIrInputDynamic) {
      GE_ASSERT_EQ(index, ir_input_2_range_iter->second.first);
      GE_ASSERT_TRUE(ir_input_2_range_iter->second.second > 0U);
      GE_ASSERT_SUCCESS(GenerateDynamicInputCode(src_nodes, index, ir_input_2_range_iter->second.second,
                                                 node_name_of_python_, input_name, output_file));
      index += ir_input_2_range_iter->second.second;
    } else {
      GE_ASSERT_TRUE(ir_input_type == ge::IrInputType::kIrInputOptional);
      if (ir_input_2_range_iter->second.second == 0U) {
        GELOGI("  optional input[%zu] has no input nodes.", ir_input_index);
      } else {
        GE_ASSERT_EQ(1U, ir_input_2_range_iter->second.second);
        const auto &src_node = src_nodes.at(index).first;
        uint32_t out_idx = src_nodes.at(index).second->GetIdx();
        GenerateInputCode(node_name_of_python_, input_name, src_node, out_idx, output_file);
        ++index;
      }
    }
  }
  return SUCCESS;
}

void PythonCodeDumper::GenerateGraphInstance(const AscGraph &asc_graph, std::ofstream &output_file) {
  output_file << "graph = ascir.HintGraph(" << "\"" << asc_graph.GetName() << "\"" << ")\n";
  for (const auto &axis:asc_graph.GetAllAxis()) {
    std::string size_name("size_of_" + axis->name);
    if (axis->size.IsConstExpr()) {
      output_file << size_name << " = ascir.SizeExpr(" << axis->size.Str().get() << ")\n";
    } else {
      output_file << size_name << " = graph.create_size(" << "\"" << axis->size.Str().get() << "\"" << ")\n";
    }
    output_file << axis->name << " = " << "graph.create_axis(" << "\"" << axis->name << "\"" << ", " << size_name
                << ")\n";
    axis_infos_.emplace_back(axis->name, size_name);
  }
}

Status PythonCodeDumper::GenerateTensorCode(const NodePtr &node, std::ofstream &output_file) {
  GELOGD("Start to gen tensor code for %s %s", node->GetNamePtr(), node->GetTypePtr());
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &ir_outputs = op_desc->GetIrOutputs();
  if (ir_outputs.size() != op_desc->GetAllOutputsDescSize()) {
    GELOGW("%s %s has output ir size %zu but has out tensor desc size %zu",
           op_desc->GetNamePtr(), op_desc->GetTypePtr(), ir_outputs.size(), op_desc->GetAllOutputsDescSize());
    return SUCCESS;
  }

  size_t output_index = 0;
  for (const auto &tensor_desc : op_desc->GetAllOutputsDescPtr()) {
    const auto &out_name = ir_outputs[output_index++].first;
    auto dtype = static_cast<ge::DataType>(tensor_desc->GetDataType());
    auto python_dtype = GenerateDataTypeCode(dtype);
    output_file << node_name_of_python_ << "." << out_name << ".dtype = " << python_dtype << std::endl;
    // todoo:后续这里增加更多的tensor信息的设置
    auto tensor_group_attr = tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_group_attr);
    if (tensor_group_attr->axis.empty()) {
      continue;
    }

    const auto &axis_code = GenerateAxisCode(tensor_group_attr->axis, axis_infos_);
    output_file << node_name_of_python_ << "." << out_name << ".axis = " << axis_code << std::endl;
    const auto &axis_size_code = GenerateAxisSizeCode(tensor_group_attr->axis, tensor_group_attr->repeats, axis_infos_);
    output_file << node_name_of_python_ << "." << out_name << ".size = " << axis_size_code << std::endl;
    const auto &axis_stride_code = GenerateAxisStrideCode(tensor_group_attr->strides);
    output_file << node_name_of_python_ << "." << out_name << ".strides = " << axis_stride_code << std::endl;
  }
  return SUCCESS;
}

void PythonCodeDumper::GenerateFooter(std::ofstream &output_file) {
  output_file << "fuser = Autofuser(AutofuserOptions())\n";
  output_file << "fused_NpuKernel0Graph = fuser.autofuse(graph)\n";
  output_file << "op_proto, tiling_def, host_impl, device_impl = fuser.codegen(graph, fused_NpuKernel0Graph)\n";
}
}
}