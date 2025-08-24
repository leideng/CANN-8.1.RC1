/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_AGING_POLICY_LRU_H_
#define GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_AGING_POLICY_LRU_H_
#include "graph/cache_policy/aging_policy.h"
#include "graph/cache_policy/policy_register.h"

namespace ge {
class AgingPolicyLru : public AgingPolicy {
public:
  virtual ~AgingPolicyLru() override = default;
  void SetDeleteInterval(const uint64_t &interval) {
    delete_interval_ = interval;
  }
  void SetCachedAgingDepth(size_t depth) override {
    (void)depth;
  }
  bool IsReadyToAddCache(const CacheHashKey hash_key, const CacheDescPtr &cache_desc) override {
    (void) hash_key;
    (void) cache_desc;
    return true;
  }
  std::vector<CacheItemId> DoAging(const CacheState &cache_state) const override;

private:
  uint64_t delete_interval_ = 0U;
};

REGISTER_AGING_POLICY_CREATOR(AgingPolicyType::AGING_POLICY_LRU,
                              []() {
                                return make_shared<AgingPolicyLru>();
                              });
}  // namespace ge
#endif
