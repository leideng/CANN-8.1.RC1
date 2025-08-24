/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#ifndef AICPU_CONTEXT_COMMON_LOG_H
#define AICPU_CONTEXT_COMMON_LOG_H

#include <sys/syscall.h>
#include <cstdint>
#include <unistd.h>

#include "toolchain/slog.h"
#include "toolchain/slog_api.h"

namespace aicpu {
inline int64_t GetTid()
{
  thread_local static int64_t tid = syscall(__NR_gettid);
  return tid;
}
}

#ifdef RUN_TEST
const char KERNEL_MODULE[] = "AICPU";
#define KERNEL_LOG_DEBUG(fmt, ...)                                    \
  printf("[DEBUG] [%s][%s][%s:%d][tid:%ld]:" fmt "\n", KERNEL_MODULE, \
         __FILE__, __FUNCTION__, __LINE__, aicpu::GetTid(), ##__VA_ARGS__)
#define KERNEL_LOG_INFO(fmt, ...)                                              \
  printf("[INFO] [%s][%s][%s:%d][tid:%ld]:" fmt "\n", KERNEL_MODULE, __FILE__, \
         __FUNCTION__, __LINE__, aicpu::GetTid(), ##__VA_ARGS__)
#define KERNEL_LOG_WARN(fmt, ...)                                              \
  printf("[WARN] [%s][%s][%s:%d][tid:%ld]:" fmt "\n", KERNEL_MODULE, __FILE__, \
         __FUNCTION__, __LINE__, aicpu::GetTid(), ##__VA_ARGS__)
#define KERNEL_LOG_ERROR(fmt, ...)                                    \
  printf("[ERROR] [%s][%s][%s:%d][tid:%ld]:" fmt "\n", KERNEL_MODULE, \
         __FILE__, __FUNCTION__, __LINE__, aicpu::GetTid(), ##__VA_ARGS__)
#define KERNEL_LOG_EVENT(fmt, ...)                                    \
  printf("[EVENT] [%s][%s][%s:%d][tid:%ld]:" fmt "\n", KERNEL_MODULE, \
         __FILE__, __FUNCTION__, __LINE__, aicpu::GetTid(), ##__VA_ARGS__)
#else
#define KERNEL_LOG_DEBUG(fmt, ...)                                                    \
  do {                                                                                \
    if (CheckLogLevel(AICPU, DLOG_DEBUG) == 1) {                                      \
      if (DlogRecord == nullptr) {                                                    \
        DlogInner(AICPU, DLOG_DEBUG, "[%s:%d][%s][tid:%ld]" fmt, __FILE__, __LINE__,  \
                  __func__, aicpu::GetTid(), ##__VA_ARGS__);                          \
      } else {                                                                        \
        DlogRecord(AICPU, DLOG_DEBUG, "[%s:%d][%s][tid:%ld]" fmt, __FILE__, __LINE__, \
                   __func__, aicpu::GetTid(), ##__VA_ARGS__);                         \
      }                                                                               \
    }                                                                                 \
  } while (0)

#define KERNEL_LOG_INFO(fmt, ...)                                                    \
  do {                                                                               \
    if (CheckLogLevel(AICPU, DLOG_INFO) == 1) {                                      \
      if (DlogRecord == nullptr) {                                                   \
        DlogInner(AICPU, DLOG_INFO, "[%s:%d][%s][tid:%ld]" fmt, __FILE__, __LINE__,  \
                  __func__, aicpu::GetTid(), ##__VA_ARGS__);                         \
      } else {                                                                       \
        DlogRecord(AICPU, DLOG_INFO, "[%s:%d][%s][tid:%ld]" fmt, __FILE__, __LINE__, \
                   __func__, aicpu::GetTid(), ##__VA_ARGS__);                        \
      }                                                                              \
    }                                                                                \
  } while (0)

#define KERNEL_LOG_WARN(fmt, ...)                                                    \
  do {                                                                               \
    if (CheckLogLevel(AICPU, DLOG_WARN) == 1) {                                      \
      if (DlogRecord == nullptr) {                                                   \
        DlogInner(AICPU, DLOG_WARN, "[%s:%d][%s][tid:%ld]" fmt, __FILE__, __LINE__,  \
                  __func__, aicpu::GetTid(), ##__VA_ARGS__);                         \
      } else {                                                                       \
        DlogRecord(AICPU, DLOG_WARN, "[%s:%d][%s][tid:%ld]" fmt, __FILE__, __LINE__, \
                   __func__, aicpu::GetTid(), ##__VA_ARGS__);                        \
      }                                                                              \
    }                                                                                \
  } while (0)

#define KERNEL_LOG_ERROR(fmt, ...)                                                  \
  do {                                                                              \
    if (DlogRecord == nullptr) {                                                    \
      DlogInner(AICPU, DLOG_ERROR, "[%s:%d][%s][tid:%ld]" fmt, __FILE__, __LINE__,  \
                __func__, aicpu::GetTid(), ##__VA_ARGS__);                          \
    } else {                                                                        \
      DlogRecord(AICPU, DLOG_ERROR, "[%s:%d][%s][tid:%ld]" fmt, __FILE__, __LINE__, \
                 __func__, aicpu::GetTid(), ##__VA_ARGS__);                         \
    }                                                                               \
  } while (0)

#define KERNEL_LOG_EVENT(fmt, ...)                                                                   \
  do {                                                                                               \
    if (DlogRecord == nullptr) {                                                                     \
      DlogInner(static_cast<int32_t>(AICPU | RUN_LOG_MASK), DLOG_EVENT, "[%s:%d][%s][tid:%ld]" fmt,  \
                __FILE__, __LINE__, __func__, aicpu::GetTid(), ##__VA_ARGS__);                       \
    } else {                                                                                         \
      DlogRecord(static_cast<int32_t>(AICPU | RUN_LOG_MASK), DLOG_EVENT, "[%s:%d][%s][tid:%ld]" fmt, \
                 __FILE__, __LINE__, __func__, aicpu::GetTid(), ##__VA_ARGS__);                      \
    }                                                                                                \
  } while (0)
#endif

#define KERNEL_CHECK_NULLPTR_VOID(value, logText...) \
  if (value == nullptr) {                            \
    KERNEL_LOG_ERROR(logText);                       \
    return;                                          \
  }

#define KERNEL_CHECK_FALSE(condition, errorCode, logText...)  \
  if (!(condition)) {                                         \
    KERNEL_LOG_ERROR(logText);                                \
    return errorCode;                                         \
  }

#define KERNEL_CHECK_NULLPTR(value, errorCode, logText...) \
  if (value == nullptr) {                                  \
    KERNEL_LOG_ERROR(logText);                             \
    return errorCode;                                      \
  }

#define KERNEL_CHECK_ASSIGN_64S_MULTI(A, B, result, errorCode)            \
  if ((A) != 0 && (B) != 0 && ((INT64_MAX) / (A)) <= (B)) {               \
    KERNEL_LOG_ERROR("Integer reversed multiA: %llu * multiB: %llu", (A), \
                     (B));                                                \
    return errorCode;                                                     \
  }                                                                       \
  (result) = ((A) * (B));

#endif  // AICPU_CONTEXT_COMMON_LOG_H
