/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "register/op_ext_gentask_registry.h"

namespace fe {
OpExtGenTaskRegistry &OpExtGenTaskRegistry::GetInstance() {
  static OpExtGenTaskRegistry registry;
  return registry;
}

OpExtGenTaskFunc OpExtGenTaskRegistry::FindRegisterFunc(const std::string &op_type) const {
  auto iter = names_to_register_func_.find(op_type);
  if (iter == names_to_register_func_.end()) {
    return nullptr;
  }
  return iter->second;
}
void OpExtGenTaskRegistry::Register(const std::string &op_type, const OpExtGenTaskFunc func) {
  names_to_register_func_[op_type] = func;
}
OpExtGenTaskRegister::OpExtGenTaskRegister(const char *op_type, OpExtGenTaskFunc func) noexcept {
  OpExtGenTaskRegistry::GetInstance().Register(op_type, func);
}
}  // namespace fe
