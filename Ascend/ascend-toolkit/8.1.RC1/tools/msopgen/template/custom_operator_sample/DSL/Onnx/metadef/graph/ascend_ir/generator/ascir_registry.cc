/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/
#include <iostream>
#include "graph/ascend_ir/ascir_registry.h"
namespace ge {
namespace ascir {
AscirRegistry &AscirRegistry::GetInstance() {
  static AscirRegistry registry;
  return registry;
}
void AscirRegistry::RegisterAscIr(const std::string &type, const AscIrDef &def) {
  types_to_ascir_[type] = def;
}
const std::unordered_map<std::string, AscIrDef> &AscirRegistry::GetAll() const {
  return types_to_ascir_;
}
}  // namespace ascir
}
