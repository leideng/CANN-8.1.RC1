/* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/
#ifndef METADEF_CXX_INC_GRAPH_ASCEND_IR_ASCEND_IR_CHECK_H_
#define METADEF_CXX_INC_GRAPH_ASCEND_IR_ASCEND_IR_CHECK_H_
#include "common/ge_common/debug/ge_log.h"
#include "common/checker.h"

namespace ge {
class AscIRException : public std::exception {
 public:
  struct Info {
    graphStatus error_code;
    std::string error_msg;
  };
  explicit AscIRException(const Info &info);
  const Info &GetInfo() const;
  const char *what() const noexcept override {
    return info_.error_msg.c_str();
  }
 private:
  Info info_;
};
}

inline bool IsVarNameValidAllowEmpty(const std::string &str) {
  if (str.empty()) {
    return true;
  }
  // 首字符必须是字母或下划线
  char first = str[0];
  if (!std::isalpha(static_cast<unsigned char>(first)) && first != '_') {
    return false;
  }

  // 后续字符只能是字母、数字或下划线
  for (size_t i = 1U; i < str.size(); ++i) {
    char c = str[i];
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
      return false;
    }
  }

  return true;
}

#define CHECK_NOTNULL_WITH_THROW_EXCEPTION(val, ...)                                        \
 ASCIR_ASSERT_NOTNULL(val, __VA_ARGS__)                                                     \

#define CHECK_VALID_IDENTIFIER_ALLOW_EMPTY_WITH_THROW_EXCEPTION(error_core, identifier)     \
  do {                                                                                      \
    if (!IsVarNameValidAllowEmpty(identifier)) {                                                      \
      GELOGE(error_core, "[Check][Identifier:" #identifier "] is invalid");                 \
      throw AscIRException({error_core, "[Check][Identifier:" #identifier "] is invalid"}); \
    }                                                                                       \
  } while (false)

#define CHECK_BOOL_WITH_THROW_EXCEPTION(error_core, val, ...)                               \
  do {                                                                                      \
    if (!(val)) {                                                                           \
      GELOGE(error_core, "[Check][Expr:" #val "] is false." __VA_ARGS__);                    \
      throw AscIRException({error_core, "[Check][Expr:" #val "] is false."});                \
    }                                                                                       \
  } while (false)

#define ASCIR_ASSERT_EQ(x, y)                                                                                             \
  do {                                                                                                                 \
    const auto &xv = (x);                                                                                              \
    const auto &yv = (y);                                                                                              \
    if (xv != yv) {                                                                                                    \
      std::stringstream ss;                                                                                            \
      ss << "Assert (" << #x << " == " << #y << ") failed, " << xv << " is not equal to " << yv;                       \
      REPORT_INNER_ERROR("E19999", "%s", ss.str().c_str());                                                            \
      GELOGE(ge::FAILED, "%s", ss.str().c_str());                                                                      \
      throw ge::AscIRException({ge::FAILED, ss.str().c_str()});                                                        \
    }                                                                                                                  \
  } while (false)

#define ASCIR_ASSERT(exp, ...) \
  do {                                                                                                                 \
    if (!(exp)) {                                                                                                      \
      auto msg = CreateErrorMsg(__VA_ARGS__);                                                                          \
      if (msg.empty()) {                                                                                               \
        REPORT_INNER_ERROR("E19999", "Assert %s failed", #exp);                                                        \
        GELOGE(ge::FAILED, "Assert %s failed", #exp);                                                                  \
        throw ge::AscIRException({ge::FAILED, #exp});                                                                  \
      } else {                                                                                                         \
        REPORT_INNER_ERROR("E19999", "%s", msg.data());                                                                \
        GELOGE(ge::FAILED, "%s", msg.data());                                                                          \
        throw ge::AscIRException({ge::FAILED, msg.data()});                                                             \
      }                                                                                                                \
    }                                                                                                                  \
  } while (false)
#define ASCIR_ASSERT_NOTNULL(v, ...) ASCIR_ASSERT(((v) != nullptr), __VA_ARGS__)
#define ASCIR_ASSERT_SUCCESS(v, ...) ASCIR_ASSERT(((v) == ge::SUCCESS), __VA_ARGS__)
#define ASCIR_ASSERT_GRAPH_SUCCESS(v, ...) ASCIR_ASSERT(((v) == ge::GRAPH_SUCCESS), __VA_ARGS__)
#define ASCIR_ASSERT_TRUE(v, ...) ASCIR_ASSERT((v), __VA_ARGS__)
#endif // METADEF_CXX_INC_GRAPH_ASCEND_IR_ASCEND_IR_CHECK_H_
