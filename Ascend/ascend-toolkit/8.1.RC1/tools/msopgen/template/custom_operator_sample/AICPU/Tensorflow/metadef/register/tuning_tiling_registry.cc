/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "register/tuning_tiling_registry.h"
#include "common/ge_common/debug/ge_log.h"

namespace tuningtiling {
ge::AscendString TuningTilingDef::GetClassName() const {
  return class_name_;
}

std::map<ge::AscendString, TuningTilingDefConstructor> &TuningTilingClassFactory::RegisterInfo() {
  static std::map<ge::AscendString, TuningTilingDefConstructor> instance;
  return instance;
}

void TuningTilingClassFactory::RegisterTilingData(const ge::AscendString &optype,
                                                  TuningTilingDefConstructor const constructor) {
  if (constructor == nullptr) {
    return;
  }
  auto &instance = TuningTilingClassFactory::RegisterInfo();
  instance[optype] = constructor;
  GELOGI("optype: %s, registered count: %zu", optype.GetString(), instance.size());
}

std::shared_ptr<TuningTilingDef> TuningTilingClassFactory::CreateTilingDataInstance(const ge::AscendString &optype) {
  const auto &instance = TuningTilingClassFactory::RegisterInfo();
  const auto it = instance.find(optype);
  if (it == instance.cend()) {
    GELOGW("can not find optype: %s", optype.GetString());
    return nullptr;
  }

  TuningTilingDefConstructor const constructor = it->second;

  if (constructor == nullptr) {
    GELOGW("CreateTilingDataInstance: constructor is nullptr");
    return nullptr;
  }

  return (*constructor)();
}
}  // namespace tuningtiling
