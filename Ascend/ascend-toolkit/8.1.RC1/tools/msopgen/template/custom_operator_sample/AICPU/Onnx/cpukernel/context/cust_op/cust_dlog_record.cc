/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include "cust_dlog_record.h"

#include <unistd.h>
#include <sys/syscall.h>
#include <cstdarg>
#include <cstdio>
#include "securec.h"

extern "C" {
__attribute__((visibility("default"))) int32_t CustSetCpuKernelContext(uint64_t workspace_size,
                                                                       uint64_t workspace_addr) {
  return aicpu::CustCpuKernelDlogUtils::GetInstance().CustSetCpuKernelContext(workspace_size, workspace_addr);
}
}

namespace {
const int32_t kMaxLogLen = 1024;
const int32_t kInvalidInput = 1;
const int32_t kCreateCtxErr = 2;
}  // namespace

namespace aicpu {

CustCpuKernelDlogUtils &CustCpuKernelDlogUtils::GetInstance() {
  static CustCpuKernelDlogUtils instance;
  return instance;
}

CustCpuKernelDlogUtils::~CustCpuKernelDlogUtils() {
  ctx_map_.clear();
}

int64_t CustCpuKernelDlogUtils::GetTid() {
  thread_local static int64_t tid = syscall(__NR_gettid);
  return tid;
}

int32_t CustCpuKernelDlogUtils::CustSetCpuKernelContext(uint64_t workspace_size, uint64_t workspace_addr) {
  if (workspace_size <= 1UL || workspace_addr == 0UL) {
    return kInvalidInput;
  }

  DeviceType device_type = DEVICE;
  CpuKernelContext *tmp = new (std::nothrow) CpuKernelContext(device_type);
  if (tmp == nullptr) {
    return kCreateCtxErr;
  }

  int64_t tid = GetTid();
  auto ctx = std::shared_ptr<aicpu::CpuKernelContext>(tmp);
  ctx->workspace_addr_ = workspace_addr;
  ctx->workspace_size_ = workspace_size;
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  if (ctx_map_.find(tid) != ctx_map_.end()) {
    return 0;
  }
  ctx_map_[tid] = ctx;
  return 0;
}

std::shared_ptr<CpuKernelContext> CustCpuKernelDlogUtils::GetCpuKernelContext() {
  int64_t tid = GetTid();
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  if (ctx_map_.find(tid) != ctx_map_.end()) {
    return ctx_map_[tid];
  }
  return nullptr;
}
}  // namespace aicpu

int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel) {
  (void)moduleId;
  (void)logLevel;
  return 1;
}

void DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...) {
  aicpu::CustCpuKernelDlogUtils &cust_dlog = aicpu::CustCpuKernelDlogUtils::GetInstance();

  auto it = cust_dlog.module_to_string_map_.find(moduleId);
  std::string module_name = (it != cust_dlog.module_to_string_map_.end()) ? it->second : "UNKNOWN_MODULE";

  std::shared_ptr<aicpu::CpuKernelContext> ctx = cust_dlog.GetCpuKernelContext();
  if (ctx == nullptr) {
    return;
  }

  va_list args;
  va_start(args, fmt);

  char formatted_msg[kMaxLogLen];
  int32_t len = snprintf_s(formatted_msg, kMaxLogLen, kMaxLogLen - 1, "[%s]", module_name.c_str());
  if (len < 0 || len >= kMaxLogLen) {
    va_end(args);
    return;
  }

  len += vsnprintf_s(formatted_msg + len, kMaxLogLen - len, kMaxLogLen - len - 1, fmt, args);
  if (len < 0 || len >= kMaxLogLen) {
    va_end(args);
    return;
  }

  switch (level) {
    case DLOG_DEBUG:
      aicpu::CustCpuKernelUtils::CustLogDebug(*ctx, formatted_msg);
      break;
    case DLOG_INFO:
      aicpu::CustCpuKernelUtils::CustLogInfo(*ctx, formatted_msg);
      break;
    case DLOG_WARN:
      aicpu::CustCpuKernelUtils::CustLogWarning(*ctx, formatted_msg);
      break;
    case DLOG_ERROR:
      aicpu::CustCpuKernelUtils::CustLogError(*ctx, formatted_msg);
      break;
    default:
      break;
  }
  va_end(args);
}