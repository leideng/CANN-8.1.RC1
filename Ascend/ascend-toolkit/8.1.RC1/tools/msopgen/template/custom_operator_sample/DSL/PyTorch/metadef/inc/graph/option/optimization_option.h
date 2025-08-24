/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_GRAPH_OPTION_OPTIMIZATION_OPTION_H_
#define INC_GRAPH_OPTION_OPTIMIZATION_OPTION_H_

#include <map>
#include <unordered_map>
#include "graph/ge_error_codes.h"
#include "optimization_option_info.h"

namespace ge {
class OptimizationOption {
 public:
  OptimizationOption() = default;
  ~OptimizationOption() = default;

  graphStatus Initialize(const std::map<std::string, std::string> &ge_options,
                         const std::unordered_map<std::string, OoInfo> &registered_options);
  graphStatus GetValue(const std::string &opt_name, std::string &opt_value) const;
  static graphStatus IsOoLevelValid(const std::string &oo_level);
  static graphStatus IsOptionValueValid(const std::string &opt_name, const std::string &opt_value,
                                        OoInfo::ValueChecker checker);

 private:
  graphStatus InitWorkingOolevel(const std::map<std::string, std::string> &ge_options);
  void PrintAllWorkingOo();

 private:
  OoLevel working_oo_level_{OoLevel::kEnd};
  std::unordered_map<std::string, std::string> working_opt_names_to_value_;
};
}  // namespace ge
#endif  //  INC_GRAPH_OPTION_OPTIMIZATION_OPTION_H_
