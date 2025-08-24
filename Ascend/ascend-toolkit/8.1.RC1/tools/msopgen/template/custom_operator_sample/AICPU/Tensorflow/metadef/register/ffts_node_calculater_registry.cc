/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "register/ffts_node_calculater_registry.h"
#include "common/hyper_status.h"

namespace gert {
FFTSNodeCalculaterRegistry &FFTSNodeCalculaterRegistry::GetInstance() {
  static FFTSNodeCalculaterRegistry registry;
  return registry;
}

FFTSNodeCalculaterRegistry::NodeCalculater FFTSNodeCalculaterRegistry::FindNodeCalculater(const string &func_name) {
  auto iter = names_to_calculater_.find(func_name);
  if (iter == names_to_calculater_.end()) {
    return nullptr;
  }
  return iter->second;
}

void FFTSNodeCalculaterRegistry::Register(const string &func_name,
                                          const FFTSNodeCalculaterRegistry::NodeCalculater func) {
  names_to_calculater_[func_name] = func;
}

FFTSNodeCalculaterRegister::FFTSNodeCalculaterRegister(const string &func_name,
    FFTSNodeCalculaterRegistry::NodeCalculater func) noexcept {
  FFTSNodeCalculaterRegistry::GetInstance().Register(func_name, func);
}
}  // namespace gert
