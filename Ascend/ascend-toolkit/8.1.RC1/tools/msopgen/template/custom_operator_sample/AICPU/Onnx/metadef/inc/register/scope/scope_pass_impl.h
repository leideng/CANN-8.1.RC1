/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef REGISTER_SCOPE_SCOPE_PASS_IMPL_H
#define REGISTER_SCOPE_SCOPE_PASS_IMPL_H

#include "external/register/scope/scope_fusion_pass_register.h"

namespace ge {
class ScopesResult::ScopesResultImpl {
 public:
  void SetScopes(const std::vector<Scope *> &scopes) { scopes_ = scopes; }
  const std::vector<Scope *> &GetScopes() const { return scopes_; }
  void SetNodes(const std::vector<ge::OperatorPtr> &nodes) { nodes_ = nodes; }
  const std::vector<ge::OperatorPtr> &GetNodes() const { return nodes_; }

 private:
  std::vector<Scope *> scopes_;  // multiple scopes
  std::vector<ge::OperatorPtr> nodes_;  // op outside of scope
};

class ScopeBasePass::ScopeBasePassImpl {
 public:
  explicit ScopeBasePassImpl(ScopeBasePass *const parent) : parent_(parent) {}
  virtual ~ScopeBasePassImpl();

  Status Run(std::shared_ptr<ScopeGraph> &scope_graph);

 private:
  Status AddFusionScopesResultToScopeGraph(const std::shared_ptr<ScopeGraph> &scope_graph,
                                           std::vector<ScopesResult> &scope_results) const;
  // Match rules one by one, support multiple sets of matching rules, and finally output a single scope
  // Note: This function does not have to be rewritten.
  //       In order to match the fusion rules designed by you better,
  //       you can implement your specific versions separately.
  bool MatchAllBatches(const ScopeTree *scope_tree, std::vector<Scope *> &results);

  bool MatchOneBatch(const ScopeTree *const scope_tree, const std::vector<ScopePattern *> &patternlist,
                     std::vector<Scope *> &results) const;
  bool MatchOneScope(const ScopePattern *pattern, Scope *scope, std::vector<Scope *> &results) const;
  Status PrintFusionScopeInfo(std::shared_ptr<ScopeGraph> &scope_graph) const;

 private:
  std::vector<ScopeFusionPatterns> patterns_;
  ScopeBasePass *parent_;
};
}  // namespace ge
#endif  // REGISTER_SCOPE_SCOPE_PASS_IMPL_H
