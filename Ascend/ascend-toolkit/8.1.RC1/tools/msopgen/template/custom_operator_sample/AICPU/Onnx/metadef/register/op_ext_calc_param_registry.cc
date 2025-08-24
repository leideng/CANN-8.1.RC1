/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "register/op_ext_calc_param_registry.h"
#include "proto/task.pb.h"

namespace fe {
OpExtCalcParamRegistry &OpExtCalcParamRegistry::GetInstance() {
  static OpExtCalcParamRegistry registry;
  return registry;
}

OpExtCalcParamFunc OpExtCalcParamRegistry::FindRegisterFunc(const std::string &op_type) const {
  auto iter = names_to_register_func_.find(op_type);
  if (iter == names_to_register_func_.end()) {
    return nullptr;
  }
  return iter->second;
}

void OpExtCalcParamRegistry::Register(const std::string &op_type, OpExtCalcParamFunc const func) {
  names_to_register_func_[op_type] = func;
}

OpExtGenCalcParamRegister::OpExtGenCalcParamRegister(const char *op_type, OpExtCalcParamFunc func) noexcept {
  OpExtCalcParamRegistry::GetInstance().Register(op_type, func);
}
} // namespace fe
