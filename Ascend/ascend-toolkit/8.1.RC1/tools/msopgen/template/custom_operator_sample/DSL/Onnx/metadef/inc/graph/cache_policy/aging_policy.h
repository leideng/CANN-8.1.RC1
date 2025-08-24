/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_AGING_POLICY_H_
#define GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_AGING_POLICY_H_
#include "graph/cache_policy/cache_state.h"

namespace ge {
constexpr const size_t kDefaultCacheQueueDepth = 1U;
class AgingPolicy {
 public:
  AgingPolicy() = default;
  virtual ~AgingPolicy() = default;
  virtual void SetCachedAgingDepth(size_t depth) = 0;
  virtual std::vector<CacheItemId> DoAging(const CacheState &cache_state) const = 0;
  virtual bool IsReadyToAddCache(const CacheHashKey hash_key, const CacheDescPtr &cache_desc) = 0;
 private:
  AgingPolicy &operator=(const AgingPolicy &anging_polocy) = delete;
  AgingPolicy(const AgingPolicy &anging_polocy) = delete;
};
}
#endif
