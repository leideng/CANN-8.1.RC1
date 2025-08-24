/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "register/node_converter_registry.h"
#include "common/hyper_status.h"

namespace gert {
NodeConverterRegistry &NodeConverterRegistry::GetInstance() {
  static NodeConverterRegistry registry;
  return registry;
}

NodeConverterRegistry::NodeConverter NodeConverterRegistry::FindNodeConverter(const string &func_name) {
  auto data = FindRegisterData(func_name);
  if (data == nullptr) {
    return nullptr;
  }
  return data->converter;
}
void NodeConverterRegistry::RegisterNodeConverter(const std::string &func_name, NodeConverter func) {
  names_to_register_data_[func_name] = {func, -1};
}
const NodeConverterRegistry::ConverterRegisterData *NodeConverterRegistry::FindRegisterData(
    const string &func_name) const {
  auto iter = names_to_register_data_.find(func_name);
  if (iter == names_to_register_data_.end()) {
    return nullptr;
  }
  return &iter->second;
}
void NodeConverterRegistry::Register(const string &func_name,
                                     const NodeConverterRegistry::ConverterRegisterData &data) {
  names_to_register_data_[func_name] = data;
}
NodeConverterRegister::NodeConverterRegister(const char *lower_func_name,
                                             NodeConverterRegistry::NodeConverter func) noexcept {
  NodeConverterRegistry::GetInstance().Register(lower_func_name, {func, -1});
}
NodeConverterRegister::NodeConverterRegister(const char *lower_func_name, int32_t require_placement,
                                             NodeConverterRegistry::NodeConverter func) noexcept {
  NodeConverterRegistry::GetInstance().Register(lower_func_name, {func, require_placement});
}
}  // namespace gert
