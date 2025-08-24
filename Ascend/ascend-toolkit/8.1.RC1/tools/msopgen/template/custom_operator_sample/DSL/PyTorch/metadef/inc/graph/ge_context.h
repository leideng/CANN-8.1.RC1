/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_GRAPH_GE_CONTEXT_H_
#define INC_GRAPH_GE_CONTEXT_H_

#include <string>
#include <map>
#include <vector>
#include "graph/ge_error_codes.h"
#include "graph/option/optimization_option.h"

namespace ge {
class GEContext {
 public:
  graphStatus GetOption(const std::string &key, std::string &option);
  const std::string &GetReadableName(const std::string &key);
  bool GetHostExecFlag() const;
  bool GetTrainGraphFlag() const;
  bool IsOverflowDetectionOpen() const;
  bool IsGraphLevelSat() const;
  uint64_t GetInputFusionSize() const;
  uint64_t SessionId() const;
  uint32_t DeviceId() const;
  int32_t StreamSyncTimeout() const;
  int32_t EventSyncTimeout() const;
  void Init();
  void SetSessionId(const uint64_t session_id);
  void SetContextId(const uint64_t context_id);
  void SetCtxDeviceId(const uint32_t device_id);
  void SetStreamSyncTimeout(const int32_t timeout);
  void SetEventSyncTimeout(const int32_t timeout);
  graphStatus SetOptionNameMap(const std::string &option_name_map_json);
  void SetMultiBatchShapeIndex(uint32_t graph_id,
      const std::map<int32_t, std::vector<int32_t>> &data_index_and_shape_map);
  const std::map<int32_t, std::vector<int32_t>> GetMultiBatchShapeIndex(uint32_t graph_id);
  OptimizationOption &GetOo() const;

 private:
  thread_local static uint64_t session_id_;
  thread_local static uint64_t context_id_;
  uint32_t device_id_ = 0U;
  // GEContext不允许拓展新的成员变量
};  // class GEContext

/// Get context
/// @return
GEContext &GetContext();
static_assert(sizeof(GEContext) == 4U, "Do not add member to a thread-safe global variable");
}  // namespace ge
#endif  //  INC_GRAPH_GE_CONTEXT_H_
