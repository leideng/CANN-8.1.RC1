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

#ifndef OPS_BUILT_IN_AICPU_CONTEXT_STUB_AICPU_CONTEXT_H
#define OPS_BUILT_IN_AICPU_CONTEXT_STUB_AICPU_CONTEXT_H

#include <sys/types.h>

#include <cstdint>
#include <string>

namespace aicpu {
typedef struct {
  uint32_t deviceId;
  uint32_t tsId;
  pid_t hostPid;
  uint32_t vfId;
} aicpuContext_t;

typedef enum {
  AICPU_ERROR_NONE = 0,
  AICPU_ERROR_FAILED = 1,
} status_t;

enum CtxType : int32_t {
  CTX_DEFAULT = 0,
  CTX_PROF,
  CTX_DEBUG
};

enum AicpuRunMode : uint32_t {
  PROCESS_PCIE_MODE = 0U,    // dc, with host mode
  PROCESS_SOCKET_MODE = 1U,  // MDC
  THREAD_MODE = 2U,          // ctrlcpu/minirc/lhisi
  INVALID_MODE = 3U,
};

const std::string CONTEXT_KEY_OP_NAME = "opname";
const std::string CONTEXT_KEY_WAIT_TYPE = "waitType";
const std::string CONTEXT_KEY_WAIT_ID = "waitId";

status_t aicpuSetContext(aicpuContext_t *ctx);

status_t __attribute__((weak)) aicpuGetContext(aicpuContext_t *ctx);

status_t InitTaskMonitorContext(uint32_t aicpuCoreCnt);

status_t SetAicpuThreadIndex(uint32_t threadIndex);

status_t __attribute__((weak)) SetOpname(const std::string &opname);

status_t GetOpname(uint32_t threadIndex, std::string &opname);

status_t __attribute__((weak)) SetThreadLocalCtx(const std::string &key, const std::string &value);

status_t __attribute__((weak)) GetThreadLocalCtx(const std::string &key, std::string &value);

status_t __attribute__((weak)) GetTaskAndStreamId(uint64_t &taskId, uint32_t &streamId);

status_t __attribute__((weak)) GetAicpuRunMode(uint32_t &run_mode); 

uint32_t __attribute__((weak)) GetUniqueVfId();

status_t RemoveThreadLocalCtx(const std::string &key);

uint32_t GetAicpuThreadIndex();
}  // namespace aicpu
#endif