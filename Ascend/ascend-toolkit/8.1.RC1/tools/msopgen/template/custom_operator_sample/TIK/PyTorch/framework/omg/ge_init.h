/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_FRAMEWORK_OMG_GE_INIT_H_
#define INC_FRAMEWORK_OMG_GE_INIT_H_
#include <map>
#include <string>
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
class GE_FUNC_VISIBILITY GEInit {
 public:
  // GE Environment Initialize, return Status: SUCCESS,FAILED
  static Status Initialize(const std::map<std::string, std::string> &options);

  static std::string GetPath();

  // GE Environment Finalize, return Status: SUCCESS,FAILED
  static Status Finalize();
};
}  // namespace ge

#endif  // INC_FRAMEWORK_OMG_GE_INIT_H_
