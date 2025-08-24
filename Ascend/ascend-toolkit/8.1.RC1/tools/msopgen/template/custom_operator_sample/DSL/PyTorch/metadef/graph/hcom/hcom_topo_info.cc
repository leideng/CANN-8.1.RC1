/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "external/hcom/hcom_topo_info.h"

#include "graph/debug/ge_log.h"
namespace ge {
Status HcomTopoInfo::SetGroupTopoInfo(const char_t *group, const HcomTopoInfo::TopoInfo &info) {
  if (group == nullptr) {
    REPORT_INNER_ERROR("E18888", "Group key is nullptr,set failed.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Group key is nullptr,set failed.");
    return GRAPH_FAILED;
  }
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    rank_info_[group] = info;
  }
  GELOGI("Add group %s successfully.", group);
  return GRAPH_SUCCESS;
}

Status HcomTopoInfo::GetGroupRankSize(const char_t *group, int64_t &rank_size) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto &iter_info = rank_info_.find(group);
    if (iter_info == rank_info_.end()) {
      REPORT_INNER_ERROR("E18888", "Group key [%s] has not been added, get failed.", group);
      GELOGE(GRAPH_FAILED, "[Check][Param] group key [%s] has not been added, get failed.", group);
      return GRAPH_FAILED;
    }
    rank_size = iter_info->second.rank_size;
  }
  return GRAPH_SUCCESS;
}

Status HcomTopoInfo::SetGroupOrderedStream(const char_t *group, void *stream) {
  if (group == nullptr) {
    REPORT_INNER_ERROR("E18888", "Group is nullptr,set failed.");
    GELOGE(GRAPH_FAILED, "[Check][Param] group is nullptr,set failed.");
    return GRAPH_FAILED;
  }
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    group_to_ordered_stream_[group] = stream;
  }
  GELOGI("Add group %s stream %p successfully.", group, stream);
  return GRAPH_SUCCESS;
}

Status HcomTopoInfo::GetGroupOrderedStream(const char_t *group, void *&stream) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto &iter_info = group_to_ordered_stream_.find(group);
    if (iter_info == group_to_ordered_stream_.end()) {
      GELOGW("[Check][Param] group [%s] has not been added, get failed.", group);
      return GRAPH_FAILED;
    }
    stream = iter_info->second;
  }

  return GRAPH_SUCCESS;
}

HcomTopoInfo::TopoDescs *HcomTopoInfo::GetGroupTopoDesc(const char_t *group) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto &iter_info = rank_info_.find(group);
  if (iter_info == rank_info_.end()) {
    REPORT_INNER_ERROR("E18888", "Group key [%s] has not been added, get failed.", group);
    GELOGE(GRAPH_FAILED, "[Check][Param] group key [%s] has not been added, get failed.", group);
    return nullptr;
  }
  return &iter_info->second.topo_level_descs;
}

Status HcomTopoInfo::GetGroupNotifyHandle(const char_t *group, void *&notify_handle) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto &iter_info = rank_info_.find(group);
    if (iter_info == rank_info_.end()) {
      REPORT_INNER_ERROR("E18888", "Group key [%s] has not been added, get failed.", group);
      GELOGE(GRAPH_FAILED, "[Check][Param] group key [%s] has not been added, get failed.", group);
      return GRAPH_FAILED;
    }
    notify_handle = iter_info->second.notify_handle;
  }
  return GRAPH_SUCCESS;
}

HcomTopoInfo &HcomTopoInfo::Instance() {
  static HcomTopoInfo hcom_topo_info;
  return hcom_topo_info;
}

bool HcomTopoInfo::TryGetGroupTopoInfo(const char_t *group, HcomTopoInfo::TopoInfo &info) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto &iter_info = rank_info_.find(group);
    if (iter_info == rank_info_.end()) {
      return false;
    }
    info = iter_info->second;
  }
  GELOGI("Get existed info of group %s successfully.", group);
  return true;
}

bool HcomTopoInfo::TopoInfoHasBeenSet(const char_t *group) {
  const std::lock_guard<std::mutex> lock(mutex_);
  return rank_info_.find(group) != rank_info_.end();
}

}
