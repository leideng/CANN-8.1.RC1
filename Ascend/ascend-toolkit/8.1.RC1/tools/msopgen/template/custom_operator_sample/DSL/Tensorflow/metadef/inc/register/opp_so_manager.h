/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_REGISTER_OPP_SO_MANAGER_H_
#define INC_REGISTER_OPP_SO_MANAGER_H_

#include "graph/ge_error_codes.h"

namespace ge {
class OppSoManager {
 public:
  OppSoManager() = default;
  ~OppSoManager() = default;
  static OppSoManager &GetInstance();
  void LoadOppPackage() const;
  void LoadOpsProtoPackage() const;
  void LoadOpMasterPackage() const;

 private:
  void LoadOpsProtoSo(bool is_split = true) const;
  void LoadOpMasterSo(bool is_split = true) const;

  void LoadUpgradedOpsProtoSo() const;
  void LoadUpgradedOpMasterSo() const;
};
}  // namespace ge

#endif  // INC_REGISTER_OPP_SO_MANAGER_H_