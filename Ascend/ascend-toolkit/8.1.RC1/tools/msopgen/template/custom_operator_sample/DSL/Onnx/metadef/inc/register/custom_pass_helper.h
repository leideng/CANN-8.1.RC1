/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_REGISTER_CUSTOM_PASS_HELPER_H_
#define INC_REGISTER_CUSTOM_PASS_HELPER_H_

#include <set>
#include "external/ge_common/ge_api_error_codes.h"
#include "external/register/register_custom_pass.h"
#include "external/register/register_types.h"

namespace ge {
class CustomPassHelper {
 public:
  static CustomPassHelper &Instance();

  void Insert(const PassRegistrationData &reg_data);

  Status Load();

  Status Unload();

  Status Run(GraphPtr &graph, CustomPassContext &custom_pass_context) const;

  Status Run(GraphPtr &graph, CustomPassContext &custom_pass_context, const CustomPassStage stage) const;

  ~CustomPassHelper() = default;

 private:
  CustomPassHelper() = default;
  std::vector<PassRegistrationData> registration_datas_;
  std::vector<void *> handles_;
};
} // namespace ge

#endif // INC_REGISTER_CUSTOM_PASS_HELPER_H_
