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
#include "device_cpu_kernel.h"

#include "aicpu_context.h"
#include "cce/aicpu_engine_struct.h"
#include "cce/fwk_adpt_struct.h"
#include "cpu_kernel_cache.h"
#include "log.h"
#include "session_cache.h"
#include "status.h"

using namespace aicpu;
namespace {
uint32_t ParseExtSessionInfo(AicpuParamHead *param_head,
                             SessionInfo *&session) {
  KERNEL_LOG_INFO("Parse extend session info begin");
  uint32_t offset = 0;
  FWKAdapter::ExtInfo *ext_info = nullptr;
  char *ext_info_addr =
      reinterpret_cast<char *>(static_cast<uintptr_t>(param_head->extInfoAddr));
  while (offset + sizeof(FWKAdapter::ExtInfo) <= param_head->extInfoLength) {
    ext_info = reinterpret_cast<FWKAdapter::ExtInfo *>(ext_info_addr + offset);
    if (ext_info == nullptr) {
      KERNEL_LOG_ERROR(
          "Extend info is nullptr, extend info length[%u], extend info "
          "offset[%u].",
          param_head->extInfoLength, offset);
      return KERNEL_STATUS_PARAM_INVALID;
    }

    if (ext_info->infoType == static_cast<int32_t>(FWKAdapter::FWK_ADPT_EXT_SESSION_INFO)) {
      auto need_len = sizeof(SessionInfo);
      if (ext_info->infoLen != need_len) {
        KERNEL_LOG_ERROR(
            "Parse extend session info failed, as info length must be "
            "[%zu], but %u.",
            sizeof(SessionInfo), ext_info->infoLen);
        return KERNEL_STATUS_PARAM_INVALID;
      }

      session = reinterpret_cast<SessionInfo *>(ext_info->infoMsg);
      KERNEL_LOG_INFO("Parse extend session info success.");
      return KERNEL_STATUS_OK;
    }

    // not overflow
    offset += FWKAdapter::kExtInfoHeadSize;
    offset += ext_info->infoLen;
  }

  KERNEL_LOG_INFO("Parse extend session info end");
  return KERNEL_STATUS_OK;
}
}  // namespace

extern "C" {
__attribute__((visibility("default"))) uint32_t RunCpuKernel(void *param) {
  KERNEL_LOG_INFO("RunCpuKernel C begin");
  if (param == nullptr) {
    KERNEL_LOG_ERROR("Param is null.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // parse param_len
  AicpuParamHead *param_head = static_cast<AicpuParamHead *>(param);
  if (param_head->length < sizeof(AicpuParamHead)) {
    KERNEL_LOG_ERROR(
        "Param length[%u] can't be less than AicpuParamHead length[%zu]",
        param_head->length, sizeof(AicpuParamHead));
    return KERNEL_STATUS_PARAM_INVALID;
  }

  SessionInfo *session = nullptr;
  uint32_t ret = ParseExtSessionInfo(param_head, session);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  if (session == nullptr) {
    KERNEL_LOG_INFO("RunCpuKernel directly.");
    CpuKernelCache cache;
    (void)cache.Init(false);
    return static_cast<uint32_t>(cache.RunKernel(param));
  }

  uint64_t task_id = 0UL;
  uint32_t stream_id = 0U;
  if (aicpu::GetTaskAndStreamId != nullptr) {
    (void)aicpu::GetTaskAndStreamId(task_id, stream_id);
  }
  return static_cast<uint32_t>(SessionCache<CpuCacheData>::Instance().RunKernel<CpuKernelCache>(
      param, session->sessionId, static_cast<uint64_t>(stream_id), session->sessFlag));
}

__attribute__((visibility("default"))) uint32_t RunCpuKernelWithBlock(void *param,
                                                                      struct BlkDimInfo *blkdim_info) {
  if (param == nullptr || blkdim_info == nullptr) {
    KERNEL_LOG_ERROR("Param is null.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  KERNEL_LOG_INFO("RunCpuKernelWithBlock C begin. blockid[%u], blockdim[%u].",
                  blkdim_info->blockId, blkdim_info->blockNum);
  // parse param_len
  AicpuParamHead *param_head = static_cast<AicpuParamHead *>(param);
  if (param_head->length < sizeof(AicpuParamHead)) {
    KERNEL_LOG_ERROR(
        "Param length[%u] can't be less than AicpuParamHead length[%zu]",
        param_head->length, sizeof(AicpuParamHead));
    return KERNEL_STATUS_PARAM_INVALID;
  }

  SessionInfo *session = nullptr;
  uint32_t ret = ParseExtSessionInfo(param_head, session);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  if (session == nullptr) {
    KERNEL_LOG_INFO("RunCpuKernelWithBlock directly.");
    CpuKernelCache cache;
    (void)cache.Init(false);
    return static_cast<uint32_t>(cache.RunCpuKernelWithBlock(param, blkdim_info));
  }

  uint64_t task_id = 0UL;
  uint32_t stream_id = 0U;
  if (aicpu::GetTaskAndStreamId != nullptr) {
    (void)aicpu::GetTaskAndStreamId(task_id, stream_id);
  }
  return static_cast<uint32_t>(SessionCache<CpuCacheData>::Instance().RunCpuKernelWithBlock<CpuKernelCache>(
      param, session->sessionId, static_cast<uint64_t>(stream_id), session->sessFlag, blkdim_info));
}
}
