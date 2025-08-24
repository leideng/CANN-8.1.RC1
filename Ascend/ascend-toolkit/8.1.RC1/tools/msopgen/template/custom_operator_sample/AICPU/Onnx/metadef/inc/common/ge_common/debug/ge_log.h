/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_COMMON_GE_COMMON_DEBUG_GE_LOG_H_
#define INC_COMMON_GE_COMMON_DEBUG_GE_LOG_H_

#include <cinttypes>
#include <cstdint>

#include "common/ge_common/ge_inner_error_codes.h"
#include "common/util/error_manager/error_manager.h"
#include "toolchain/slog.h"
#ifdef __GNUC__
#include <unistd.h>
#include <sys/syscall.h>
#else
#include "mmpa/mmpa_api.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GE_MODULE_NAME static_cast<int32_t>(GE)
#define GE_MODULE_NAME_U16 static_cast<uint16_t>(GE)

// trace status of log
enum TraceStatus { TRACE_INIT = 0, TRACE_RUNNING, TRACE_WAITING, TRACE_STOP };

class GE_FUNC_VISIBILITY GeLog {
 public:
  static uint64_t GetTid() {
#ifdef __GNUC__
    const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
#else
    const uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
#endif
    return tid;
  }
};

inline bool IsLogEnable(const int32_t module_name, const int32_t log_level) {
  const int32_t enable = CheckLogLevel(module_name, log_level);
  // 1:enable, 0:disable
  return (enable == 1);
}

inline bool IsLogPrintStdout() {
  static int32_t stdout_flag = -1;
  if (stdout_flag == -1) {
    const char *env_ret = getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    const bool print_stdout = ((env_ret != nullptr) && (strcmp(env_ret, "1") == 0));
    stdout_flag = print_stdout ? 1 : 0;
  }
  return (stdout_flag == 1) ? true : false;
}

#define GELOGE(ERROR_CODE, fmt, ...)                                                                \
  do {                                                                                              \
    dlog_error(GE_MODULE_NAME, "%" PRIu64 " %s: ErrorNo: %" PRIuLEAST8 "(%s) %s" fmt, \
	       GeLog::GetTid(), &__FUNCTION__[0U], \
               (ERROR_CODE), ((GE_GET_ERRORNO_STR(ERROR_CODE)).c_str()),                            \
               ErrorManager::GetInstance().GetLogHeader().c_str(), ##__VA_ARGS__);                  \
  } while (false)

#define GELOGW(fmt, ...)                                                                          \
  do {                                                                                            \
    dlog_warn(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
  } while (false)

#define GELOGI(fmt, ...)                                                                          \
  do {                                                                                            \
    dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
  } while (false)

#define GELOGD(fmt, ...)                                                                           \
  do {                                                                                             \
    dlog_debug(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
  } while (false)

#define GEEVENT(fmt, ...)                                                                        \
  do {                                                                                                               \
    dlog_info(static_cast<int32_t>(static_cast<uint32_t>(RUN_LOG_MASK) | static_cast<uint32_t>(GE_MODULE_NAME)),     \
        "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__);                            \
    if (!IsLogPrintStdout()) {                                                \
      dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
    }                                                                                            \
  } while (false)

#define GERUNINFO(fmt, ...)                                                                        \
  do {                                                                                                               \
    dlog_info(static_cast<int32_t>(static_cast<uint32_t>(RUN_LOG_MASK) | static_cast<uint32_t>(GE_MODULE_NAME)),     \
        "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__);                            \
    if (!IsLogPrintStdout()) {                                                \
      dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
    }                                                                                            \
  } while (false)

#define GELOGT(VALUE, fmt, ...)                                                                                        \
  do {                                                                                                                 \
    constexpr const char_t *TraceStatStr[] = {"INIT", "RUNNING", "WAITING", "STOP"};                                   \
    constexpr int32_t idx = static_cast<int32_t>(VALUE);                                                               \
    char_t *v = const_cast<char_t *>(TraceStatStr[idx]);                                                               \
    dlog_info((static_cast<uint32_t>(RUN_LOG_MASK) | static_cast<uint32_t>(GE_MODULE_NAME)),                           \
              "[status:%s]%" PRIu64 " %s:" fmt, v, GeLog::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__);                 \
  } while (false)

#define GE_LOG_ERROR(MOD_NAME, ERROR_CODE, fmt, ...)                                                           \
  do {                                                                                                         \
    dlog_error((MOD_NAME), "%" PRIu64 " %s: ErrorNo: %" PRIuLEAST8 "(%s) %s" fmt, GeLog::GetTid(), \
	       &__FUNCTION__[0U], (ERROR_CODE),  \
               ((GE_GET_ERRORNO_STR(ERROR_CODE)).c_str()), ErrorManager::GetInstance().GetLogHeader().c_str(), \
               ##__VA_ARGS__);                                                                                 \
  } while (false)

// print memory when it is greater than 1KB.
#define GE_PRINT_DYNAMIC_MEMORY(FUNC, PURPOSE, SIZE)                                                        \
  do {                                                                                                      \
    if (static_cast<size_t>(SIZE) > 1024UL) {                                                               \
      GELOGI("MallocMemory, func=%s, size=%zu, purpose=%s", (#FUNC), static_cast<size_t>(SIZE), (PURPOSE)); \
    }                                                                                                       \
  } while (false)

#define GELOG_DEPRECATED(option)                                                                                \
  do {                                                                                                          \
    std::cout << "[WARNING][GE] Option " << (option) << " is deprecated and will be removed in future version." \
                 " Please do not configure this option in the future." << std::endl;                            \
  } while (false)


#ifdef __cplusplus
}
#endif
#endif  // INC_COMMON_GE_COMMON_DEBUG_GE_LOG_H_
