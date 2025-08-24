/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_GRAPH_PARALLELISM_COMM_TASK_BUILDER_H_
#define METADEF_GRAPH_PARALLELISM_COMM_TASK_BUILDER_H_

#include "graph/parallelism/tensor_parallel_attrs.h"
#include "nlohmann/json.hpp"

namespace ge {
namespace tp {
class CommTaskBuilder {
 public:
  static CommTaskBuilder &GetInstance() {
    static CommTaskBuilder instance;
    return instance;
  }

  void BuildCommTask(const nlohmann::json &j, CommTask &comm_task);
  Status ConvertToJson(const CommTask &comm_task, nlohmann::json &j);

 private:
  CommTaskBuilder();
  ~CommTaskBuilder() = default;

  void InitCommTaskBuilders();
  void InitJsonConverters();
  template<typename T>
  static Status ConvertToJson(const T *reshard_task, nlohmann::json &j);

  std::map<std::string, std::function<void(const nlohmann::json &, CommTask &)>> builders_;
  std::map<std::string, std::function<Status(const CommTask &, nlohmann::json &)>> json_converters_;
};
}  // namespace tp
}  // namespace ge

#endif  // METADEF_GRAPH_PARALLELISM_COMM_TASK_BUILDER_H_
