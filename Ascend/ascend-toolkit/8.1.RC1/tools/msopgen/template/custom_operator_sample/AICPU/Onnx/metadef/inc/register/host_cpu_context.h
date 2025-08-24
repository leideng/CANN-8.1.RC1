/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_REGISTER_HOST_CPU_CONTEXT_H_
#define INC_REGISTER_HOST_CPU_CONTEXT_H_

#include "external/ge_common/ge_api_error_codes.h"
#include "register/register_types.h"

namespace ge {
class HostCpuContext {
 public:
  HostCpuContext() = default;
  ~HostCpuContext() = default;
 private:
  class Impl;
  Impl *impl_;
};
} // namespace ge

#endif // INC_REGISTER_HOST_CPU_CONTEXT_H_
