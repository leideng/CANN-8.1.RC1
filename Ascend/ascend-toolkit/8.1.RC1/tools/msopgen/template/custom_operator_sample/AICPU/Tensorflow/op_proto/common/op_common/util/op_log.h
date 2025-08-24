/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file op_log.h
 * \brief
 */

#ifndef OP_COMMON_OP_LOG_H_
#define OP_COMMON_OP_LOG_H_

#include <type_traits>
#include <string>
#include <cstdint>
#include "toolchain/slog.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/node.h"
#include "graph/op_desc.h"

#define OPPROTO_SUBMOD_NAME "OP_COMMON"
#define OP 1

namespace opcommon {
inline const char* GetCstr(const std::string& str) {
  return str.c_str();
}

inline const char* GetCstr(const char* str) {
  return str;
}

inline const char* GetOpInfo(const char* str) {
  return str;
}

template <class T>
constexpr bool IsContextType() {
  return !std::is_base_of<ge::Operator, typename std::decay<T>::type>::value &&
      !std::is_same<const char*, typename std::decay<T>::type>::value;
}

template <class T>
typename std::enable_if<IsContextType<T>(), std::string>::type GetOpInfo(T context) {
  if (context == nullptr) {
    return "nil:nil";
  }
  std::string opInfo = context->GetNodeType() != nullptr ? context->GetNodeType() : "nil";
  std::string opName = context->GetNodeName() != nullptr ? context->GetNodeName() : "nil";
  opInfo += ":";
  opInfo += opName;
  return opInfo;
}

#define OP_LOG_SUB(moduleId, submodule, level, fmt, ...)                                                   \
  do {                                                                                                  \
    if (OpCheckLogLevel(static_cast<int32_t>(moduleId), static_cast<int32_t>(level)) == 1) {             \
        OpDlogInner(moduleId, level, "[%s:%d][%s]" fmt, __FILE__, __LINE__, submodule, ##__VA_ARGS__);    \
    }                                                                                                   \
  } while (0)

#define COMMON_OP_LOG_SUB(moduleId, level, OpInfo, fmt, ...)                            \
  OP_LOG_SUB(moduleId, OPPROTO_SUBMOD_NAME, level, " %s:%d OpName:[%s]" #fmt, __FUNCTION__, \
          __LINE__, GetCstr(OpInfo), ##__VA_ARGS__)

#define OP_LOGI(opname, ...) D_OP_LOGI(GetOpInfo(opname), __VA_ARGS__)
#define OP_LOGW(opname, ...) D_OP_LOGW(GetOpInfo(opname), __VA_ARGS__)

#define OP_LOGE_WITHOUT_REPORT(opname, ...) D_OP_LOGE(GetOpInfo(opname), __VA_ARGS__)
#define OP_LOGE(opname, ...)                       \
  do {                                              \
    OP_LOGE_WITHOUT_REPORT(opname, ##__VA_ARGS__); \
    REPORT_INNER_ERROR("EZ9999", ##__VA_ARGS__);    \
  } while (0)

#define OP_LOGD(opname, ...) D_OP_LOGD(GetOpInfo(opname), __VA_ARGS__)

#define D_OP_LOGI(opname, fmt, ...) COMMON_OP_LOG_SUB(OP, DLOG_INFO, opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGW(opname, fmt, ...) COMMON_OP_LOG_SUB(OP, DLOG_WARN, opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGE(opname, fmt, ...) COMMON_OP_LOG_SUB(OP, DLOG_ERROR, opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGD(opname, fmt, ...) COMMON_OP_LOG_SUB(OP, DLOG_DEBUG, opname, fmt, ##__VA_ARGS__)

int32_t OpCheckLogLevel(int32_t moduleId, int32_t logLevel);
void OpDlogInner(int moduleId, int level, const char *fmt, ...);
} // namespace opcommon
#endif // OP_COMMON_OP_LOG_H_