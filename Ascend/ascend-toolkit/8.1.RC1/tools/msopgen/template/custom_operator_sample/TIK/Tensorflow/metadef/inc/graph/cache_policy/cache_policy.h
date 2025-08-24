/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GRAPH_CACHE_POLICY_CACHE_POLICY_H_
#define GRAPH_CACHE_POLICY_CACHE_POLICY_H_

#include <vector>
#include <memory>
#include "cache_state.h"
#include "policy_register.h"
#include "graph/ge_error_codes.h"

namespace ge {
class CachePolicy {
 public:
  ~CachePolicy() = default;

  CachePolicy(const CachePolicy &) = delete;
  CachePolicy(CachePolicy &&) = delete;
  CachePolicy &operator=(const CachePolicy &) = delete;
  CachePolicy &operator=(CachePolicy &&) = delete;

  static std::unique_ptr<CachePolicy> Create(const MatchPolicyPtr &mp, const AgingPolicyPtr &ap);
  static std::unique_ptr<CachePolicy> Create(const MatchPolicyType mp_type, const AgingPolicyType ap_type,
                                             size_t cached_aging_depth = kDefaultCacheQueueDepth);

  graphStatus SetMatchPolicy(const MatchPolicyPtr mp);

  graphStatus SetAgingPolicy(const AgingPolicyPtr ap);

  CacheItemId AddCache(const CacheDescPtr &cache_desc);

  CacheItemId FindCache(const CacheDescPtr &cache_desc) const;

  std::vector<CacheItemId> DeleteCache(const DelCacheFunc &func);

  std::vector<CacheItemId> DeleteCache(const std::vector<CacheItemId> &delete_item);

  std::vector<CacheItemId> DoAging();

  CachePolicy() = default;

 private:
  CacheState compile_cache_state_;
  MatchPolicyPtr mp_ = nullptr;
  AgingPolicyPtr ap_ = nullptr;
};
}  // namespace ge
#endif
