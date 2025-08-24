/* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef AUTOFUSE_ASCEND_GRAPH_CODE_DUMPER_H
#define AUTOFUSE_ASCEND_GRAPH_CODE_DUMPER_H

#include "graph/utils/code_dumper/code_dumper_base.h"
namespace ge {
namespace ascir {
class PythonCodeDumper : public CodeDumperBase<PythonCodeDumper, AscGraph> {
 public:
  static void GenerateHeader(std::ofstream &output_file);
  void GenerateGraphInstance(const AscGraph &asc_graph, std::ofstream &output_file);
  static void GenerateFooter(std::ofstream &output_file);
  Status GenerateNodeCode(const NodePtr &node, std::ofstream &output_file);
  Status GenerateDataEdgeCode(const Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>> &src_nodes,
                              const NodePtr &dst_node,
                              std::ofstream &output_file);
  Status GenerateTensorCode(const NodePtr &node, std::ofstream &output_file);
 private:
  // axis name: axis size
  std::vector<std::pair<std::string, std::string>> axis_infos_;
  std::string node_name_of_python_;
};

// C++代码生成器
class CppCodeDumper : public CodeDumperBase<CppCodeDumper, AscGraph> {
 public:
};
}
}
#endif // AUTOFUSE_ASCEND_GRAPH_CODE_DUMPER_H

