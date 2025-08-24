/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "graph/cache_policy/match_policy_exact_only.h"
namespace ge {
CacheItemId MatchPolicyExactOnly::GetCacheItemId(const CCStatType &cc_state, const CacheDescPtr &desc) const {
  const CacheHashKey hash_key = desc->GetCacheDescHash();
  const auto &iter = cc_state.find(hash_key);
  if (iter == cc_state.end()) {
    GELOGD("can not find without shape hash %lu", hash_key);
    return KInvalidCacheItemId;
  }
  const auto &info_vec = iter->second;
  const auto cached_info = std::find_if(info_vec.begin(), info_vec.end(), [&desc] (const CacheInfo &cached) {
      return (cached.GetCacheDesc()->IsMatch(desc));
  });
  if (cached_info != info_vec.end()) {
    return cached_info->GetItemId();
  } else {
    return KInvalidCacheItemId;
  }
}
}
