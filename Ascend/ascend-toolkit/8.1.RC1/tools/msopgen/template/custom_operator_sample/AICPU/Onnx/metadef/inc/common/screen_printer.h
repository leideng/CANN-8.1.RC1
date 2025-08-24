/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_INC_COMMON_SCREEN_PRINTER_H_
#define METADEF_INC_COMMON_SCREEN_PRINTER_H_

#include <mutex>
#include <string>
#include <unordered_set>
#include <unistd.h>
#include "external/graph/types.h"

namespace ge {
class ScreenPrinter {
 public:
  static ScreenPrinter &GetInstance();
  void Log(const char *fmt, ...);
  void Init(const std::string &print_mode);

 private:
  ScreenPrinter() = default;
  ~ScreenPrinter() = default;

  ScreenPrinter(const ScreenPrinter &) = delete;
  ScreenPrinter(const ScreenPrinter &&) = delete;
  ScreenPrinter &operator=(const ScreenPrinter &)& = delete;
  ScreenPrinter &operator=(const ScreenPrinter &&)& = delete;

  enum class PrintMode : uint32_t {
    ENABLE = 0U,
    DISABLE = 1U
  };
  PrintMode print_mode_ = PrintMode::ENABLE;
  std::mutex mutex_;
};

#define SCREEN_LOG(fmt, ...)                                       \
  do {                                                             \
    ScreenPrinter::GetInstance().Log(fmt, ##__VA_ARGS__);          \
  } while (false)
}  // namespace ge
#endif  // METADEF_INC_COMMON_SCREEN_PRINTER_H_
