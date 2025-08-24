/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef AIR_CXX_BASE_COMMON_HYPER_STATUS_H_
#define AIR_CXX_BASE_COMMON_HYPER_STATUS_H_
#include <memory>
#include <cstdarg>
#include "graph/types.h"
namespace gert {
ge::char_t *CreateMessage(const ge::char_t *format, va_list arg);
class HyperStatus {
 public:
  bool IsSuccess() const {
    return status_ == nullptr;
  }
  const ge::char_t *GetErrorMessage() const noexcept {
    return status_;
  }
  ~HyperStatus() {
    delete[] status_;
  }

  HyperStatus() {}
  HyperStatus(const HyperStatus &other);
  HyperStatus(HyperStatus &&other) noexcept;
  HyperStatus &operator=(const HyperStatus &other);
  HyperStatus &operator=(HyperStatus &&other) noexcept;

  static HyperStatus Success();
  static HyperStatus ErrorStatus(const ge::char_t *message, ...);
  static HyperStatus ErrorStatus(std::unique_ptr<ge::char_t[]> message);

 private:
  ge::char_t *status_ = nullptr;
};
}  // namespace gert

namespace ge {
using HyperStatus = gert::HyperStatus;
}  // namespace ge
#endif  // AIR_CXX_BASE_COMMON_HYPER_STATUS_H_
