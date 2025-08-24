/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef OPS_BUILT_IN_AICPU_CONTEXT_STUB_AICPU_ASYNC_EVENT_H_
#define OPS_BUILT_IN_AICPU_CONTEXT_STUB_AICPU_ASYNC_EVENT_H_
#include "aicpu_context.h"
namespace aicpu {
// inc/aucpu_schedule/aicpu_sharder/aicpu_async_event.h
struct AsyncNotifyInfo {
  uint8_t waitType;
  uint32_t waitId;
  uint64_t taskId;
  uint32_t streamId;
  uint32_t retCode;
  aicpu::aicpuContext_t ctx;
};
}  // namespace aicpu
#endif  // OPS_BUILT_IN_AICPU_CONTEXT_STUB_AICPU_ASYNC_EVENT_H_
