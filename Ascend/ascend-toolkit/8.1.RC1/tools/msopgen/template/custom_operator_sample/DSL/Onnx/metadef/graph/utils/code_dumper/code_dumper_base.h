/* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_CXX_GRAPH_UTILS_CODE_DUMPER_CODE_DUMPER_BASE_H_
#define METADEF_CXX_GRAPH_UTILS_CODE_DUMPER_CODE_DUMPER_BASE_H_
#include <string>
#include <fstream>
#include <iostream>
#include <functional>
#include "common/checker.h"
#include "graph/compute_graph.h"
#include "ascend_ir/ascend_ir_core/ascend_ir.h"

namespace ge {
// CRTP基类, Graph类型可以是ge::ComputeGraph或者ge::AscGraph
template<typename Derived, typename GraphType>
class CodeDumperBase {
 public:
  Status Dump(const GraphType &graph, const std::string &out_file_path) {
    std::ofstream output_file(out_file_path);
    GE_ASSERT_TRUE(output_file.is_open(), "out_file_path %s is invalid", out_file_path.c_str());
    static_cast<Derived *>(this)->GenerateHeader(output_file);
    static_cast<Derived *>(this)->GenerateGraphInstance(graph, output_file);
    for (const auto &node: graph.GetAllNodes()) {
      GELOGD("Start to gen code for %s %s", node->GetNamePtr(), node->GetTypePtr());
      GE_ASSERT_SUCCESS(static_cast<Derived *>(this)->GenerateNodeCode(node, output_file));
      const auto &input_nodes = node->GetInDataNodesAndAnchors();
      GE_ASSERT_SUCCESS(static_cast<Derived *>(this)->GenerateDataEdgeCode(input_nodes, node, output_file));
      GE_ASSERT_SUCCESS(static_cast<Derived *>(this)->GenerateTensorCode(node, output_file));
    }

    static_cast<Derived *>(this)->GenerateFooter(output_file);
    output_file.close();
    return SUCCESS;
  }
};
}
#endif // METADEF_CXX_GRAPH_UTILS_CODE_DUMPER_CODE_DUMPER_BASE_H_
