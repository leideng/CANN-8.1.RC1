/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include <cstdio>
#include <sys/time.h>
#include <unistd.h>

#include <cstdarg>
#include <ctime>
#include <map>
#include <mutex>
#include <string>

#include "cpu_context.h"
#include "cust_cpu_utils.h"
#include "securec.h"
namespace {
constexpr char CUST_AICPU_OP[] = "CUST_AICPU_OP";
constexpr int64_t TIME_THOUSAND_MS = 1000;
constexpr int32_t COMPUTER_BEGINER_YEAR = 1900;
constexpr uint32_t CUST_MSG_LENGTH = 1024;
constexpr uint32_t TIMESTAMP_LEN = 128;
using LOG_LEVEL = uint32_t;
constexpr uint32_t DEBUG = 0U;
constexpr uint32_t INFO = 1U;
constexpr uint32_t WARNING = 2U;
constexpr uint32_t ERROR = 3U;
static std::map<LOG_LEVEL, std::string> LEVEL_NAME = {
    {DEBUG, "DEBUG"}, {INFO, "INFO"}, {WARNING, "WARNING"}, {ERROR, "ERROR"}};

inline void *ValueToPtr(const uint64_t value) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(value));
}
}  // namespace
namespace aicpu {
int ConstructBaseLog(char *msg, uint32_t level, unsigned int msgLen,
                     const char *timestamp) {
  int err = snprintf_s(msg, msgLen, msgLen - 1, "[%s] %s:%s ",
                       LEVEL_NAME[level].c_str(), CUST_AICPU_OP, timestamp);
  return err;
}

void GetTime(char *time_str, unsigned int size) {
  // sync time zone
  tzset();
  struct timeval time_val;
  int ret = gettimeofday(&time_val, nullptr);
  if (ret != 0) {
    return;
  }
  struct tm time_info;
  struct tm *temp = localtime_r(&time_val.tv_sec, &time_info);
  if (temp == nullptr) {
    return;
  }
  int err = snprintf_s(
      time_str, size, size - 1, "%04d-%02d-%02d-%02d:%02d:%02d.%03ld.%03ld",
      time_info.tm_year + COMPUTER_BEGINER_YEAR, time_info.tm_mon + 1,
      time_info.tm_mday, time_info.tm_hour, time_info.tm_min, time_info.tm_sec,
      time_val.tv_usec / TIME_THOUSAND_MS, time_val.tv_usec % TIME_THOUSAND_MS);
  if (err != 0) {
    return;
  }
}

void CustCpuKernelUtils::SafeWrite(aicpu::CpuKernelContext &ctx, char *msg,
                                   size_t len) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  if (ctx.workspace_size_ <= 0) {
    return;
  }

  if (ctx.workspace_size_ >= len) {
    auto mem_ret = memcpy_s(ValueToPtr(ctx.workspace_addr_),
                            ctx.workspace_size_, msg, len);
    if (mem_ret != EOK) {
      return;
    }
    ctx.workspace_addr_ = ctx.workspace_addr_ + len;
    ctx.workspace_size_ -= len;
  } else {
    // 如果剩余空间不够，则只copy部分日志，并且最后一位是'\0'
    auto mem_ret =
        memcpy_s(ValueToPtr(ctx.workspace_addr_), ctx.workspace_size_ - 1, msg,
                 ctx.workspace_size_ - 1);
    if (mem_ret != EOK) {
      return;
    }
    ctx.workspace_size_ = 0;
  }
}

void CustCpuKernelUtils::WriteCustLog(aicpu::CpuKernelContext &ctx,
                                      uint32_t level, const char *fmt,
                                      va_list v) {
  char msg[CUST_MSG_LENGTH] = {0};
  char timestamp[TIMESTAMP_LEN] = {0};
  GetTime(timestamp, TIMESTAMP_LEN);
  int err = ConstructBaseLog(msg, level, CUST_MSG_LENGTH, timestamp);
  if (err == -1) {
    // 日志作为最后屏障，不再处理这种ERROR
    return;
  }

  auto len = strlen(msg);
  err = vsnprintf_truncated_s(msg + len, CUST_MSG_LENGTH - len, fmt, v);
  if (err == -1) {
    return;
  }
  msg[CUST_MSG_LENGTH - 1] = '\0';
  len = strlen(msg);
  // 如果用户没有输入换行符，我们需要给用户拼接换行符
  if (len > 1 && msg[len - 1] != '\n') {
    // msg 的最后一位不是换行符，则将'\0'替换为'\n',并将'\0'后移，
    if (len < (CUST_MSG_LENGTH - 1)) {
      msg[len] = '\n';  // 2 for \n.
      msg[len + 1] = '\0';
      len++;
    } else {
      msg[CUST_MSG_LENGTH - 2] = '\n'; // 2 is for the penultimate
    }
  }
  SafeWrite(ctx, msg, len + 1);
}

void CustCpuKernelUtils::CustLogDebug(aicpu::CpuKernelContext &ctx,
                                      const char *fmt, ...) {
  // 当workspace size不足之后，直接退出, 至少要预留'\0'
  if (ctx.workspace_size_ <= 1 || ctx.workspace_addr_ == 0UL) {
    return;
  }
  if (fmt != nullptr) {
    va_list list;
    va_start(list, fmt);
    WriteCustLog(ctx, DEBUG, fmt, list);
    va_end(list);
  }
}
void CustCpuKernelUtils::CustLogInfo(aicpu::CpuKernelContext &ctx,
                                     const char *fmt, ...) {
  if (ctx.workspace_size_ <= 1 || ctx.workspace_addr_ == 0UL) {
    return;
  }
  if (fmt != nullptr) {
    va_list list;
    va_start(list, fmt);
    WriteCustLog(ctx, INFO, fmt, list);
    va_end(list);
  }
}
void CustCpuKernelUtils::CustLogWarning(aicpu::CpuKernelContext &ctx,
                                        const char *fmt, ...) {
  if (ctx.workspace_size_ <= 1 || ctx.workspace_addr_ == 0UL) {
    return;
  }
  if (fmt != nullptr) {
    va_list list;
    va_start(list, fmt);
    WriteCustLog(ctx, WARNING, fmt, list);
    va_end(list);
  }
}
void CustCpuKernelUtils::CustLogError(aicpu::CpuKernelContext &ctx,
                                      const char *fmt, ...) {
  if (ctx.workspace_size_ <= 1 || ctx.workspace_addr_ == 0UL) {
    return;
  }
  if (fmt != nullptr) {
    va_list list;
    va_start(list, fmt);
    WriteCustLog(ctx, ERROR, fmt, list);
    va_end(list);
  }
}
}  // namespace aicpu