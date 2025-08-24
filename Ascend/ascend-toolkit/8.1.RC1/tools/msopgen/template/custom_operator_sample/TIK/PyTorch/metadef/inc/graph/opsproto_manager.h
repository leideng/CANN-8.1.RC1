/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_GRAPH_OPSPROTO_MANAGER_H_
#define INC_GRAPH_OPSPROTO_MANAGER_H_

#include <map>
#include <string>
#include <vector>
#include <mutex>

namespace ge {
class OpsProtoManager {
 public:
  OpsProtoManager() = default;
  ~OpsProtoManager();
  static OpsProtoManager *Instance();

  bool Initialize(const std::map<std::string, std::string> &options);
  void Finalize();

 private:
  void LoadOpsProtoPluginSo(const std::string &path);

  std::string pluginPath_;
  std::vector<void *> handles_;
  bool is_init_ = false;
  std::mutex mutex_;
};
}  // namespace ge

#endif  // INC_GRAPH_OPSPROTO_MANAGER_H_
