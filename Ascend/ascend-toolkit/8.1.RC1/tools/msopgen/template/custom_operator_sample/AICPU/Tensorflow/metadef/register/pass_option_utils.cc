/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
* ===================================================================================================================*/

#include "register/pass_option_utils.h"
#include "register/optimization_option_registry.h"
#include "ge_common/debug/ge_log.h"
#include "common/checker.h"
#include "graph/ge_local_context.h"

namespace ge {
graphStatus PassOptionUtils::CheckIsPassEnabled(const std::string &pass_name, bool &is_enabled) {
  std::vector<std::string> opt_names;
  const auto ret = PassOptionRegistry::GetInstance().FindOptionNamesByPassName(pass_name, opt_names);
  if (ret != SUCCESS) {
    // 若Pass未被注册，返回非零错误码，由调用方判断如何处理
    GELOGI("The pass [%s] is not registered", pass_name.c_str());
    return ret;
  }

  // 当前最多支持两级开关，opt_names.size() <= 2
  std::vector<const OoInfo *> opt_infos;
  for (const auto &opt_name : opt_names) {
    const auto info_ptr = OptionRegistry::GetInstance().FindOptInfo(opt_name);
    if (info_ptr == nullptr) {
      GELOGW("Option [%s] of pass [%s] is not registered", opt_name.c_str(), pass_name.c_str());
      continue;
    }
    opt_infos.emplace_back(info_ptr);
  }
  // Pass关联的选项均未注册,说明注册阶段遗漏了选项
  if (opt_infos.empty()) {
    GELOGW("the pass [%s] has no registered option", pass_name.c_str());
    return GRAPH_FAILED;
  }

  is_enabled = false;
  const auto &oo = GetThreadLocalContext().GetOo();
  for (auto it = opt_infos.crbegin(); it != opt_infos.crend(); ++it) {
    const auto opt = *it;
    std::string opt_value;
    if (oo.GetValue(opt->name, opt_value) == GRAPH_SUCCESS) {
      if (opt_value.empty() || (opt_value == "true")) {
        GELOGD("the pass [%s] is enabled, option [%s] is [%s]", pass_name.c_str(), opt->name.c_str(),
               opt_value.c_str());
        is_enabled = true;
        return GRAPH_SUCCESS;
      } else {
        GELOGD("the pass [%s] is disabled, option [%s] is [%s]", pass_name.c_str(), opt->name.c_str(),
               opt_value.c_str());
        return GRAPH_SUCCESS;
      }
    }
  }
  // OoTable中没有配置该Pass的开关选项，说明不使能
  GELOGD("the pass [%s] is disabled, option is not in working option table", pass_name.c_str());
  return GRAPH_SUCCESS;
}
}  // namespace ge