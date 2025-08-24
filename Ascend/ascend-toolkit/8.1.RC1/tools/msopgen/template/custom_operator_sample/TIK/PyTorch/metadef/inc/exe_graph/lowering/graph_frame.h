/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef AIR_CXX_RUNTIME_V2_METADEF_EXE_GRAPH_GRAPH_FRAME_H_
#define AIR_CXX_RUNTIME_V2_METADEF_EXE_GRAPH_GRAPH_FRAME_H_
#include <stack>
#include <memory>
#include <utility>
#include <string>
#include "graph/node.h"
#include "buffer_pool.h"
#include "bg_kernel_context_extend.h"
#include "graph/fast_graph/execute_graph.h"

namespace gert {
namespace bg {
class ValueHolder;
using ValueHolderPtr = std::shared_ptr<ValueHolder>;
constexpr const ge::char_t *kStageIdsToLastPartitionedCall = "StageIdsToLastPartitionedCall";
constexpr const ge::char_t *kStageIdsToFirstPartitionedCall = "StageIdsToFirstPartitionedCall";
/*
 * 执行的阶段, 越小越靠前执行，越大越靠后执行
 */
enum class OnMainRootLastExecStage {
  kFirstStage = 0,
  kLastEventSyncStage,
  kLastResourceClean,
  // add level before this
  kStageSize
};

/*
 * 执行的阶段, 越小越靠前执行，越大越靠后执行
 */
enum class OnMainRootFirstExecStage {
  kFirstEventSyncStage = 0,
  // add level before this
  kStageSize
};

class GraphFrame {
 public:
  GraphFrame(const GraphFrame &) = delete;
  GraphFrame(GraphFrame &&) = delete;
  GraphFrame operator=(const GraphFrame &) = delete;
  GraphFrame operator=(GraphFrame &&) = delete;

  GraphFrame(ge::ExecuteGraphPtr exe_graph, const GraphFrame &parent_frame) noexcept
      : execute_graph_(std::move(exe_graph)),
        current_compute_node_and_index_(), root_frame_(parent_frame.root_frame_),
        nodes_to_index_(root_frame_.nodes_to_index_), indexes_to_node_(root_frame_.indexes_to_node_),
        relevant_input_node_(root_frame_.relevant_input_node_) {}

  explicit GraphFrame(ge::ExecuteGraphPtr exe_graph) noexcept
      : execute_graph_(std::move(exe_graph)),
        current_compute_node_and_index_(), root_frame_(*this),
        nodes_to_index_holder_(), nodes_to_index_(nodes_to_index_holder_), indexes_to_node_holder_(),
        indexes_to_node_(indexes_to_node_holder_), relevant_input_node_holder_(),
        relevant_input_node_(relevant_input_node_holder_) {}

  const ge::NodePtr &GetCurrentComputeNode() const {
    return current_compute_node_and_index_.first;
  }
  void SetCurrentComputeNode(const ge::NodePtr &current_node) {
    if (current_node == nullptr) {
      current_compute_node_and_index_ = {nullptr, 0};
      return;
    }
    const auto result = nodes_to_index_.emplace(current_node, nodes_to_index_.size());
    current_compute_node_and_index_ = {current_node, result.first->second};
    if (result.second) {
      indexes_to_node_.emplace_back(current_node);
    }
  }
  void AddRelevantInputNode(const ge::NodePtr &current_node) {
    relevant_input_node_.emplace_back(current_node);
  }
  bool GetCurrentNodeIndex(size_t &index) const {
    if (current_compute_node_and_index_.first == nullptr) {
      return false;
    }
    index = current_compute_node_and_index_.second;
    return true;
  }

  bool IsRootFrame() const {
    return &root_frame_ == this;
  }

  const ge::ExecuteGraphPtr &GetExecuteGraph() const {
    return execute_graph_;
  }

  const vector<ge::NodePtr> &GetIndexesToNode() const {
    return indexes_to_node_;
  }

  const std::unordered_map<ge::NodePtr, size_t> &GetNodesToIndex() const {
    return nodes_to_index_;
  }

  const std::vector<ValueHolderPtr> &GetLastExecNodes() const {
    return last_exec_nodes_;
  }

  /*
   * set last exec node, its priority is first level
   */
  void SetLastExecNode(const ValueHolderPtr last_exec_node) {
    if (last_exec_node != nullptr) {
      last_exec_nodes_.emplace_back(last_exec_node);  // todo to be deprecated
    }
  }

 private:
  ge::ExecuteGraphPtr execute_graph_;
  std::pair<ge::NodePtr, size_t> current_compute_node_and_index_;
  GraphFrame &root_frame_;
  std::unordered_map<ge::NodePtr, size_t> nodes_to_index_holder_;
  std::unordered_map<ge::NodePtr, size_t> &nodes_to_index_;
  std::vector<ge::NodePtr> indexes_to_node_holder_;
  std::vector<ge::NodePtr> &indexes_to_node_;
  std::vector<ValueHolderPtr> last_exec_nodes_; // todo to be deprecated
  std::vector<ge::NodePtr> relevant_input_node_holder_;
  std::vector<ge::NodePtr> &relevant_input_node_;
};
}  // namespace bg
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_METADEF_EXE_GRAPH_GRAPH_FRAME_H_
