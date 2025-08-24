/* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_CXX_ASC_GRAPH_UTILS_H
#define METADEF_CXX_ASC_GRAPH_UTILS_H

#include "ascend_ir/ascend_ir_core/ascend_ir.h"
#include "proto/ascend_ir.pb.h"

namespace ge {
class AscGraphUtils {
 public:
  static ComputeGraphPtr GetComputeGraph(const AscGraph &asc_graph);
  /**
 * 拷贝构造AscGraph
 * @param compute_graph的node对象是Node类型时候，接口内部转换为AscNode并构造AscGraph
 * @return
 */
  static AscGraph ConvertComputeGraphToAscGraph(const ComputeGraphPtr &compute_graph);
  static graphStatus SerializeToBinary(const AscGraph &asc_graph, std::string &output);
  static graphStatus SerializeToReadable(const AscGraph &asc_graph, std::string &output);
  static graphStatus SerializeToProto(const AscGraph &asc_graph, ascend_ir::proto::AscGraphDef &asc_graph_def);
  static graphStatus DeserializeFromBinary(const std::string &to_be_deserialized, AscGraph &out_asc_graph);
  static graphStatus DeserializeFromReadable(const std::string &to_be_deserialized, AscGraph &out_asc_graph);
  static graphStatus DeserializeFromProto(const ascend_ir::proto::AscGraphDef &asc_graph_def, AscGraph &asc_graph);
};
class AscNodeSerializeUtils {
 public:
  static graphStatus SerializeIrDef(const AscNode &node, ascend_ir::proto::IrDef &ir_def);
  static graphStatus SerializeAttrGroupsDef(const AscNode &node,
                                            ascend_ir::proto::AscNodeAttrGroupsDef &asc_node_attr_groups_def);
};

class AscNodeDeserializeUtils {
 public:
  static graphStatus DeserializeIrDef(const ascend_ir::proto::IrDef &ir_def, AscNode &node);
  static graphStatus DeserializeAttrGroupsDef(const ascend_ir::proto::AscNodeAttrGroupsDef &asc_node_attr_groups_def,
                                              AscNode &node);
};
class ExpressionSerializer : public GeIrAttrSerializer {
 public:
  ExpressionSerializer() = default;
  graphStatus Serialize(const AnyValue &av, proto::AttrDef &def) override;
  graphStatus Deserialize(const proto::AttrDef &def, AnyValue &av) override;
};
}  // namespace ge

#endif  // METADEF_CXX_ASC_GRAPH_UTILS_H
