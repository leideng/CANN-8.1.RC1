/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "graph/option/optimization_option.h"
#include "common/util/error_manager/error_manager.h"
#include "common/ge_common/debug/ge_log.h"
#include "common/ge_common/string_util.h"
#include "common/checker.h"

namespace ge {
namespace {
const std::unordered_map<std::string, OoLevel> kOptValToLevels{
    {"O1", OoLevel::kO1},
    {"O3", OoLevel::kO3},
};

inline void ReportParamInvalid(const std::string &opt_name, const std::string &opt_value, const std::string &reason) {
  REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                     std::vector<std::string>({opt_name, opt_value, reason}));
  GELOGE(GRAPH_PARAM_INVALID, "[Oo][Check] the value [%s] of option[%s] is invalid. %s", opt_value.c_str(),
         opt_name.c_str(), reason.c_str());
}
}  // namespace

graphStatus OptimizationOption::Initialize(const std::map<std::string, std::string> &ge_options,
                                           const std::unordered_map<std::string, OoInfo> &registered_options) {
  working_oo_level_ = OoLevel::kEnd;
  working_opt_names_to_value_.clear();
  // 1. Initialize OoLevel if possible
  if (InitWorkingOolevel(ge_options) != GRAPH_SUCCESS) {
    return GRAPH_PARAM_INVALID;
  }
  // 2. Expand optimization template
  for (const auto &opt_info : registered_options) {
    if (OoInfoUtils::IsBitSet(opt_info.second.levels, static_cast<uint32_t>(working_oo_level_))) {
      const auto value_str = OoInfoUtils::GetDefaultValue(opt_info.second, working_oo_level_);
      (void) working_opt_names_to_value_.emplace(opt_info.first, value_str);
    }
  }
  // 3. Verify user-configured optimization options
  for (const auto &ge_opt : ge_options) {
    const auto iter = registered_options.find(ge_opt.first);
    if (iter == registered_options.cend()) {
      continue;
    }
    if (IsOptionValueValid(ge_opt.first, ge_opt.second, iter->second.checker) != GRAPH_SUCCESS) {
      return GRAPH_PARAM_INVALID;
    }
    working_opt_names_to_value_[ge_opt.first] = ge_opt.second;
  }

  PrintAllWorkingOo();
  GELOGI("Init optimization option success");
  return GRAPH_SUCCESS;
}

graphStatus OptimizationOption::GetValue(const std::string &opt_name, std::string &opt_value) const {
  const auto iter = working_opt_names_to_value_.find(opt_name);
  if (iter == working_opt_names_to_value_.cend()) {
    return GRAPH_FAILED;
  }
  opt_value = iter->second;
  return GRAPH_SUCCESS;
}

graphStatus OptimizationOption::IsOoLevelValid(const std::string &oo_level) {
  const auto &oo_level_iter = kOptValToLevels.find(oo_level);
  if (oo_level_iter == kOptValToLevels.end()) {
    ReportParamInvalid(OO_LEVEL, oo_level, "OoLevel is unsupported");
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

graphStatus OptimizationOption::IsOptionValueValid(const std::string &opt_name, const std::string &opt_value,
                                                   OoInfo::ValueChecker checker) {
  if (checker == nullptr) {
    return GRAPH_SUCCESS;
  }
  GE_ASSERT_TRUE(checker(opt_value), "Check option value failed, option [%s], value [%s]", opt_name.c_str(),
                 opt_value.c_str());
  return GRAPH_SUCCESS;
}

graphStatus OptimizationOption::InitWorkingOolevel(const std::map<std::string, std::string> &ge_options) {
  const auto opt_iter = ge_options.find(OO_LEVEL);
  if (opt_iter == ge_options.end()) {
    // default oo_level is O3 if ge_option is not set
    working_oo_level_ = OoLevel::kO3;
  } else {
    if (IsOoLevelValid(opt_iter->second) != GRAPH_SUCCESS) {
      return GRAPH_PARAM_INVALID;
    }
    working_oo_level_ = kOptValToLevels.at(opt_iter->second);
  }
  GELOGI("[Oo][Print]working_oo_level is %u.", working_oo_level_);
  return GRAPH_SUCCESS;
}

void OptimizationOption::PrintAllWorkingOo() {
  for (const auto &iter : working_opt_names_to_value_) {
    GELOGD("[Oo][Print]the value[%s] of option[%s] set successfully", iter.second.c_str(), iter.first.c_str());
  }
}
}  // namespace ge
