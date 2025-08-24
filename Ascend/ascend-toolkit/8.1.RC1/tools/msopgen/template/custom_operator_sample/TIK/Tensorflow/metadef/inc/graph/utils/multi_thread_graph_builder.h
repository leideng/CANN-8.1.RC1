/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef __INC_METADEF_MULTI_THREAD_GRAPH_BUILDER_H
#define __INC_METADEF_MULTI_THREAD_GRAPH_BUILDER_H

#include <memory>
#include <vector>
#include <mutex>
#include "external/graph/graph.h"
#include "graph/utils/graph_thread_pool.h"

namespace ge {
class MultiThreadGraphBuilder {
 public:
  explicit MultiThreadGraphBuilder(int32_t thread_num);
  ~MultiThreadGraphBuilder() = default;

  Graph &SetInputs(const std::vector<ge::Operator> &inputs, ge::Graph &graph);

 private:
  static graphStatus GetGraphRelatedOperators(const std::vector<Operator> &inputs,
                                              std::vector<OperatorImplPtr> &related_ops);
  static void GetOutputLinkOps(const OperatorImplPtr &op_impl,
                               std::vector<OperatorImplPtr> &output_op_impls);
  static graphStatus WalkForwardOperators(const std::vector<OperatorImplPtr> &vec_ops,
                                          std::vector<OperatorImplPtr> &related_ops);
  void ResetOpSubgraphBuilder(const OpDescPtr &op_desc, OperatorImplPtr &op_impl);

  int32_t thread_num_;
  std::mutex mutex_;
  std::unique_ptr<GraphThreadPool> pool_;
};
} // namespace ge
#endif // __INC_METADEF_MULTI_THREAD_GRAPH_BUILDER_H
