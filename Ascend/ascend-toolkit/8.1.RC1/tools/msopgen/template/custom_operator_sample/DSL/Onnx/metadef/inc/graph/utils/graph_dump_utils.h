/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_GRAPH_UTILS_GRAPH_DUMP_UTILS_H
#define INC_GRAPH_UTILS_GRAPH_DUMP_UTILS_H

#include "graph/compute_graph.h"
#include "graph/fast_graph/execute_graph.h"
#include "graph/utils/execute_graph_adapter.h"
#include "graph/utils/graph_utils.h"
#include "mmpa/mmpa_api.h"

namespace ge {
namespace {
const char_t *const kDumpGeGraph = "DUMP_GE_GRAPH";
}
/**
 * 将ComputeGraph落盘成文件
 * @param compute_graph 要落盘的对象
 * @param name 落盘的文件名，会拼接上默认的前后缀
 * @return
 */
inline void DumpGraph(const ComputeGraphPtr &compute_graph, const char_t *const name) {
  ge::GraphUtils::DumpGEGraph(compute_graph, name);
  ge::GraphUtils::DumpGEGraphToOnnx(*compute_graph, name);
  uint64_t i = 0U;
  for (const auto &sub_graph_func : compute_graph->GetAllSubgraphs()) {
    const auto sub_graph_func_name = std::string(name) + std::string("_sub_graph_") + std::to_string(i++);
    ge::GraphUtils::DumpGEGraph(sub_graph_func, sub_graph_func_name);
    ge::GraphUtils::DumpGEGraphToOnnx(*sub_graph_func, sub_graph_func_name);
  }
}

/**
 * 将ExecuteGraph落盘成文件
 * @param execute_graph 要落盘的对象
 * @param name 落盘的文件名，会拼接上默认的前后缀
 * @return
 */
inline void DumpGraph(ExecuteGraph *execute_graph, const char_t *const name) {
  char_t dump_ge_graph[MMPA_MAX_PATH] = {'\0'};
  if (mmGetEnv(kDumpGeGraph, &(dump_ge_graph[0U]), static_cast<uint32_t>(MMPA_MAX_PATH)) != EN_OK) {
    return;
  }
  const auto compute_graph = ExecuteGraphAdapter::ConvertExecuteGraphToComputeGraph(execute_graph);
  if (compute_graph != nullptr) {
    DumpGraph(compute_graph, name);
  }
}
}  // namespace ge
#endif  // INC_GRAPH_UTILS_GRAPH_DUMP_UTILS_H
