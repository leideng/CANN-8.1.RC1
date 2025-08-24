//===--------- tunnel_impl.h - CCE Print Header File ---------*- C++-*-===//
//
// Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file defines the two function in host for CCE.
//
//
//===----------------------------------------------------------------------===//
#ifndef CCELIB_PRINT_TUNNEL_IMPL_H
#define CCELIB_PRINT_TUNNEL_IMPL_H

#include "tunnel.h"

namespace cce {
namespace internal {
namespace DebugTunnel {
DebugTunnelData *__attribute__((weak)) Open(unsigned BlockNum) {
  DebugTunnelData debugTunnelDataForHost;
  PrintPayload::OnHostInitialize(&(debugTunnelDataForHost.PrintData), BlockNum);
  void *Hbm_debugTunnerlData_start_addr = NULL;
  rtError_t error =
      rtMalloc(reinterpret_cast<void **>(&Hbm_debugTunnerlData_start_addr),
               sizeof(debugTunnelDataForHost), RT_MEMORY_HBM);
  if (error != RT_ERROR_NONE) {
    printf("The memory for the printing function on the device side fails to "
           "be allocated.");
    printf("As a result, the printing function fails!\n");
    return nullptr;
  }
  if (Hbm_debugTunnerlData_start_addr == nullptr) {
    // failed to allocate log region
    printf("WARNING: failed to allocate DebugTunnelData memory\n");
    return nullptr;
  }
  error = rtMemcpy(Hbm_debugTunnerlData_start_addr,
                   sizeof(debugTunnelDataForHost), &debugTunnelDataForHost,
                   sizeof(debugTunnelDataForHost), RT_MEMCPY_HOST_TO_DEVICE);
  if (error != RT_ERROR_NONE) {
    printf("ERROR:The memory copy of the device print on fails,");
    printf("and the printing function is invalid!");
    return nullptr;
  }
  return reinterpret_cast<DebugTunnelData *>(Hbm_debugTunnerlData_start_addr);
}

void __attribute__((weak)) Close(DebugTunnelData *DTData, void *&Stream) {
  if (!DTData) {
    return;
  }
  DebugTunnelData debugTunnelDataForHost;
  rtError_t error = rtStreamSynchronize(Stream);
  if (error != RT_ERROR_NONE) {
    printf("ERROR:Synchronous waiting for the device print failed.");
    printf("The printing function is invalid!");
    return;
  }
  error =
      rtMemcpy(&debugTunnelDataForHost, sizeof(debugTunnelDataForHost), DTData,
               sizeof(debugTunnelDataForHost), RT_MEMCPY_DEVICE_TO_HOST);
  if (error != RT_ERROR_NONE) {
    printf("ERROR:The memory copy of the device print on fails,");
    printf("and the printing function is invalid!");
    return;
  }
  PrintPayload::OnHostFinish(&(debugTunnelDataForHost.PrintData), Stream);

  error = rtFree(DTData);
  if (error != RT_ERROR_NONE) {
    printf("ERROR:The memory free of the device print fails,");
    printf("and the device print is invalid!");
  }
}

[aicore] void __attribute__((weak))
OnKernelInitialize(__gm__ DebugTunnelData *DTData) {
  int64_t *FixStackObj =
      static_cast<int64_t *>(__builtin_cce_get_fix_stack_object());
  if (!DTData) {
    *FixStackObj = 0;
    return;
  }
  PrintPayload::OnKernelInitialize(&(DTData->PrintData));
  *FixStackObj = reinterpret_cast<int64_t>(DTData);
}

[aicore] void __attribute__((weak))
OnKernelFinish(__gm__ DebugTunnelData *DTData) {
  if (!DTData) {
    return;
  }
  PrintPayload::OnKernelFinish(&(DTData->PrintData));
}

static [aicore] __gm__ DebugTunnelData *GetKernelInstance() {
  int64_t *FixStackObj =
      static_cast<int64_t *>(__builtin_cce_get_fix_stack_object());
  if (!*FixStackObj) {
    return nullptr;
  }
  __gm__ DebugTunnelData *DTData = (__gm__ DebugTunnelData *)(*FixStackObj);
  return DTData;
}
} // namespace DebugTunnel
} // namespace internal
} // namespace cce

extern "C" {
cce::internal::DebugTunnelData *__attribute__((weak))
__DebugTunnel_Open(uint32_t BlockNum) {
  return cce::internal::DebugTunnel::Open(BlockNum);
}

void __attribute__((weak))
__DebugTunnel_Close(cce::internal::DebugTunnelData *DTData, void *&stream) {
  return cce::internal::DebugTunnel::Close(DTData, stream);
}

[aicore] void __attribute__((weak))
__DebugTunnel_Initialize(__gm__ cce::internal::DebugTunnelData *DTData) {
  cce::internal::DebugTunnel::OnKernelInitialize(DTData);
}

[aicore] void __attribute__((weak))
__DebugTunnel_Finish(__gm__ cce::internal::DebugTunnelData *DTData) {
  cce::internal::DebugTunnel::OnKernelFinish(DTData);
}
}
#endif