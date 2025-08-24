/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "inc/graph/ascend_ir/utils/ascendir_dump_utils.h"
namespace ge {
std::stringstream &DumpAscirGraph::TilingKeyStr(std::stringstream &ss, AscGraph &graph) {
  std::string Tilingkey = std::to_string(graph.GetTilingKey());
  ss << "TilingKey: " << Tilingkey << std::endl;
  return ss;
}

std::stringstream &DumpAscirGraph::NameStr(std::stringstream &ss, AscGraph &graph) {
  ss << "Graph Name: " << graph.GetName() << std::endl;
  return ss;
}

std::stringstream &DumpAscirGraph::AllAxisStr(std::stringstream &ss, AscGraph &graph) {
  ss << "Axis:" << std::endl;
  if (graph.GetAllAxis().empty()) {
    return ss;
  }
  static const char *axis_type_str[] = {
      "ORIGINAL",
      "BLOCK_OUTER",
      "BLOCK_INNER",
      "TILE_OUTER",
      "TILE_INNER",
      "MERGED",
      "INVALID",
  };
  int32_t i = 1;
  for (const auto &axis : graph.GetAllAxis()) {
    ss << "    axis"<< i << ": " << std::endl;
    ss << "        name: " << axis->name << std::endl;
    ss << "        id: " << axis->id << std::endl;
    if (axis->type >= ge::Axis::kAxisTypeOriginal && axis->type <= ge::Axis::kAxisTypeMerged) {
      ss << "        type: " << axis_type_str[axis->type] << std::endl;
    }
    std::string bind_block = axis->bind_block? "true" : "false";
    ss << "        bind_block: " << bind_block << std::endl;
    ss << "        size: " << axis->size.Str().get() << std::endl;
    ss << "        align: " << axis->align << std::endl;
    if (!axis->from.empty()) {
      ss << "        from: {";
      for (auto from_axis : axis->from) {
        ss << from_axis << ", ";
      }
      ss << "}" << std::endl;
    }
    if (axis->split_pair_other_id == kIdNone) {
      ss << "        split_pair_other_id: -1" << std::endl;
    }
    else {
      ss << "        split_pair_other_id: " << axis->split_pair_other_id << std::endl;
    }
    ss << "        allow_oversize_axis: " << axis->allow_oversize_axis << std::endl;
    ss << "        allow_unaligned_tail: " << axis->allow_oversize_axis << std::endl;
    i++;
  }
  return ss;
}

std::string DumpAscirGraph::ApiTypeToString(ge::ApiType type) {
  static const std::map<ge::ApiType, std::string> api_type_to_string_map = {
    {ge::ApiType::kAPITypeBuffer, "BUFFER"},
    {ge::ApiType::kAPITypeCompute, "COMPUTE"},
    {ge::ApiType::kAPITypeInvalid, "INVALID"},
  };
  const auto it = api_type_to_string_map.find(type);
  if (it != api_type_to_string_map.end()) {
    return it->second;
  }
  return "UNDEFINED";
}

std::string DumpAscirGraph::ComputUnitToString(ge::ComputeUnit unit) {
  static const std::map<ge::ComputeUnit, std::string> comput_unit_to_string_map = {
    {ge::ComputeUnit::kUnitNone, "NONE"},
    {ge::ComputeUnit::kUnitMTE1, "MTE1"},
    {ge::ComputeUnit::kUnitMTE2, "MTE2"},
    {ge::ComputeUnit::kUnitMTE3, "MTE3"},
    {ge::ComputeUnit::kUnitScalar, "SCALAR"},
    {ge::ComputeUnit::kUnitVector, "VECTOR"},
    {ge::ComputeUnit::kUnitCube, "CUBE"},
    {ge::ComputeUnit::kUnitInvalid, "INVALID"},
  };
  const auto it = comput_unit_to_string_map.find(unit);
  if (it != comput_unit_to_string_map.end()) {
    return it->second;
  }
  return "UNDEFINED";
}

std::string DumpAscirGraph::ComputeTypeToString(ge::ComputeType type) {
  static const std::map<ge::ComputeType, std::string> comput_type_to_string_map = {
    {ge::ComputeType::kComputeLoad, "LOAD"},
    {ge::ComputeType::kComputeStore, "STORE"},
    {ge::ComputeType::kComputeReduceStore, "REDUCE_STORE"},
    {ge::ComputeType::kComputeElewise, "ELEWISE"},
    {ge::ComputeType::kComputeBroadcast, "BROADCAST"},
    {ge::ComputeType::kComputeReduce, "REDUCE"},
    {ge::ComputeType::kComputeTranspose, "TRANPOSE"},
    {ge::ComputeType::kComputeInvalid, "INVALID"},
  };
  const auto it = comput_type_to_string_map.find(type);
  if (it != comput_type_to_string_map.end()) {
    return it->second;
  }
  return "UNDEFINED";
}

std::stringstream &DumpAscirGraph::AscNodeAttrStr(std::stringstream &ss, AscNodeAttr &attr) {
  ss << "            AscNode: " << std::endl;
  ss << "                sched: " << std::endl;
  ss << "                    exec_order: " << attr.sched.exec_order << std::endl;
  ss << "                    axis: ";
  for (auto axis : attr.sched.axis) {
    ss << axis << ", ";
  }
  ss << std::endl;
  ss << "                    loop_axis: " << attr.sched.loop_axis << std::endl;
  ss << std::endl;
  ss << "                Api: " << std::endl;
  ss << "                    Api type: " << ApiTypeToString(attr.api.type) << std::endl;
  ss << "                    Compute unit: " << ComputUnitToString(attr.api.unit) << std::endl;
  ss << "                    Compute type: " << ComputeTypeToString(attr.api.compute_type) << std::endl;
  return ss;
}

std::stringstream &DumpAscirGraph::AscTensorAttrStr(std::stringstream &ss, AscTensorAttr *attr) {
  if (attr == nullptr) {
    return ss;
  }
  ss << "            AscTensor: " << std::endl;
  ss << "                DataType: " << TypeUtils::DataTypeToSerialString(attr->dtype.operator ge::DataType()) << std::endl;
  ss << "                axis: ";
  for (auto axis : attr->axis) {
    ss << axis << ", ";
  }
  ss << std::endl;
  ss << "                repeats: ";
  for (const auto &repeat : attr->repeats) {
    ss << repeat.Str().get() << ", ";
  }
  ss << std::endl;
  ss << "                strides: ";
  for (const auto &stride : attr->strides) {
    ss << stride.Str().get() << ", ";
  }
  ss << std::endl;
  ss << "                vectorized_axis: ";
  for (auto axis : attr->vectorized_axis) {
    ss << axis << ", ";
  }
  ss << std::endl;
  ss << "                vectorized_strides: ";
  for (const auto &stride : attr->vectorized_strides) {
    ss << stride.Str().get() << ",";
  }
  ss << std::endl;
  MemAttrStr(ss, attr);
  MemQueueAttrStr(ss, attr);
  MemBufAttrStr(ss, attr);
  MemOptAttrStr(ss, attr);
  return ss;
}

std::string DumpAscirGraph::AllocTypeToString(ge::AllocType type) {
  static const std::map<ge::AllocType, std::string> alloc_type_to_string_map = {
    {ge::AllocType::kAllocTypeGlobal, "GLOBAL"},
    {ge::AllocType::kAllocTypeL1, "L1"},
    {ge::AllocType::kAllocTypeL2, "L2"},
    {ge::AllocType::kAllocTypeBuffer, "BUFFER"},
    {ge::AllocType::kAllocTypeQueue, "QUEUE"}
  };
  const auto it =  alloc_type_to_string_map.find(type);
  if (it != alloc_type_to_string_map.end()) {
    return it->second;
  }
  return "UNDEFINED";
}

std::string DumpAscirGraph::PositionToString(ge::Position position) {
  static const std::map<ge::Position, std::string> position_to_string_map = {
    {ge::Position::kPositionGM, "GM"},
    {ge::Position::kPositionVecIn, "VECIN"},
    {ge::Position::kPositionVecOut, "VECOUT"}
  };
  const auto it =  position_to_string_map.find(position);
  if (it != position_to_string_map.end()) {
    return it->second;
  }
  return "UNDEFINED";
}

std::string DumpAscirGraph::HardwareToString(ge::MemHardware hardware) {
  static const std::map<ge::MemHardware, std::string> hard_ware_to_string_map = {
    {ge::MemHardware::kMemHardwareGM, "GM"},
    {ge::MemHardware::kMemHardwareUB, "UB"}
  };
  const auto it =  hard_ware_to_string_map.find(hardware);
  if (it != hard_ware_to_string_map.end()) {
    return it->second;
  }
  return "UNDEFINED";
}

std::stringstream &DumpAscirGraph::MemAttrStr(std::stringstream &ss, AscTensorAttr *attr) {
  ss << "                MemAttr: " << std::endl;
  ss << "                    tensor_id: " << attr->mem.tensor_id << std::endl;
  ss << "                    alloc_type: " << AllocTypeToString(attr->mem.alloc_type) << std::endl;
  ss << "                    position: " << PositionToString(attr->mem.position) << std::endl;
  ss << "                    hardware: " << HardwareToString(attr->mem.hardware) << std::endl;
  ss << "                    buf_ids: ";
  for (auto buf_id : attr->mem.buf_ids) {
    ss << buf_id << ", ";
  }
  ss << std::endl;
  ss << "                    name: " << attr->mem.name << std::endl;
  return ss;
}

std::stringstream &DumpAscirGraph::MemQueueAttrStr(std::stringstream &ss, AscTensorAttr *attr) {
  ss << "                MemQueAttr: " << std::endl;
  ss << "                    id: " << attr->que.id << std::endl;
  ss << "                    depth: " << attr->que.depth << std::endl;
  ss << "                    buf_num: " << attr->que.buf_num << std::endl;
  ss << "                    name: " << attr->que.name << std::endl;
  return ss;
}

std::stringstream &DumpAscirGraph::MemBufAttrStr(std::stringstream &ss, AscTensorAttr *attr) {
  ss << "                MemBufAttr: " << std::endl;
  ss << "                    id: " << attr->buf.id << std::endl;
  ss << "                    name: " << attr->buf.name << std::endl;
  return ss;
}

std::stringstream &DumpAscirGraph::MemOptAttrStr(std::stringstream &ss, AscTensorAttr *attr) {
  ss << "                MemOptAttr: " << std::endl;
  ss << "                    reuse_id: " << attr->opt.reuse_id << std::endl;
  ss << "                    ref_tensor: " << attr->opt.ref_tensor << std::endl;
  ss << "                    merge_scope: " << attr->opt.merge_scope << std::endl;
  return ss;
}

std::stringstream &DumpAscirGraph::NodesStr(std::stringstream &ss, ge::AscNodeVisitor &nodes) {
  ss << "nodes:" << std::endl;
  int32_t i = 1;
  for (auto node = nodes.begin(); node != nodes.end(); ++node) {
    ss << "    node" << i << " info: " << std::endl;
    ss << "        node name: " << node.operator*()->GetName() << std::endl;
    uint32_t input_size = node.operator*()->inputs.Size();
    ss << "        inputs: " << std::endl;
    for (uint32_t j = 0; j < input_size; j++) {
      if ((node.operator*()->GetInDataAnchor(j) != nullptr) && (node.operator*()->GetInDataAnchor(j)->GetPeerOutAnchor() != nullptr)) {
        AscTensorAttr &temp = node.operator*()->inputs[j].attr;
        AscTensorAttr *tempPtr = &temp;
        AscTensorAttrStr(ss, tempPtr);
      }
    }
    ss << "        outputs: " << std::endl;
    for (auto outputs : node.operator*()->outputs()) {
      AscTensorAttrStr(ss, &outputs->attr);
    }
    ss << "        attr: " << std::endl;
    AscNodeAttrStr(ss, node.operator*()->attr);
    ss << std::endl;
    i++;
  }
  return ss;
}

std::string DumpAscirGraph::DumpGraph(AscGraph &graph) {
  std::stringstream ss;
  TilingKeyStr(ss, graph);
  NameStr(ss, graph);
  AllAxisStr(ss, graph);
  AscNodeVisitor all_nodes = graph.GetAllNodes();
  NodesStr(ss, all_nodes);
  return ss.str();
}

void DumpAscirGraph::WriteOutToFile(const std::string &filename, AscGraph &graph) {
  const auto &content = DumpGraph(graph);
  std::ofstream outFile(filename);
  if (!outFile) {
    std::cerr << "Cannot open the file: " << filename << std::endl;
    return;
  }
  outFile << content;
  outFile.close();
}

} // namespace ge