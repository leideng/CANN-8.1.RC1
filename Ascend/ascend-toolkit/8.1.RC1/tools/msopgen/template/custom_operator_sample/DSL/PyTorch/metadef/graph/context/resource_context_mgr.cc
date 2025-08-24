/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "graph/resource_context_mgr.h"

namespace ge {
ResourceContext *ResourceContextMgr::GetResourceContext(const std::string &resource_key) {
  const std::lock_guard<std::mutex> lk(ctx_mu_);
  const std::map<std::string, std::unique_ptr<ge::ResourceContext>>::const_iterator
      iter = resource_keys_to_contexts_.find(resource_key);
  if (iter == resource_keys_to_contexts_.cend()) {
    return nullptr;
  }
  return resource_keys_to_contexts_[resource_key].get();
}

graphStatus ResourceContextMgr::SetResourceContext(const std::string &resource_key, ResourceContext *const context) {
  const std::lock_guard<std::mutex> lk(ctx_mu_);
  resource_keys_to_contexts_[resource_key] = std::unique_ptr<ResourceContext>(context);
  return GRAPH_SUCCESS;
}

graphStatus ResourceContextMgr::RegisterNodeReliedOnResource(const std::string &resource_key, NodePtr &node) {
  const std::lock_guard<std::mutex> lk(ctx_mu_);
  (void)resource_keys_to_read_nodes_[resource_key].emplace(node);
  return GRAPH_SUCCESS;
}

OrderedNodeSet &ResourceContextMgr::MutableNodesReliedOnResource(const std::string &resource_key) {
  const std::lock_guard<std::mutex> lk(ctx_mu_);
  return resource_keys_to_read_nodes_[resource_key];
}

graphStatus ResourceContextMgr::ClearContext() {
  const std::lock_guard<std::mutex> lk_resource(ctx_mu_);
  resource_keys_to_contexts_.clear();
  resource_keys_to_read_nodes_.clear();
  return GRAPH_SUCCESS;
}
}  // namespace ge
