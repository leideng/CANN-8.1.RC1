//===--------- common_impl.h - CCE Print Header File ---------*- C++-*-===//
//
// Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file Declare the runtime API.
//
//
//===----------------------------------------------------------------------===//
#ifndef CCELIB_PRINT_DEVICE_PRINT_RUNTIME_H
#define CCELIB_PRINT_DEVICE_PRINT_RUNTIME_H

#define RT_ERROR_NONE ((uint32_t)0x0)
#define RT_MEMORY_HBM ((uint32_t)0x2)
#define RT_MEMCPY_HOST_TO_DEVICE ((uint32_t)0x1)
#define RT_MEMCPY_DEVICE_TO_HOST ((uint32_t)0x2)

typedef uint32_t rtError_t;
extern "C" uint32_t rtMalloc(void **devPtr, uint64_t size, uint32_t type);
extern "C" uint32_t rtMallocHost(void **hostPtr, uint64_t size);
extern "C" uint32_t rtMemcpyAsync(void *dst, uint64_t destMax, const void *src,
                                  uint64_t count, uint32_t kind, void *stream);
extern "C" uint32_t rtStreamSynchronize(void *stream);
extern "C" uint32_t rtFree(void *devPtr);
extern "C" uint32_t rtFreeHost(void *hostPtr);
extern "C" uint32_t rtMemcpy(void *dst, uint64_t destMax, const void *src,
                             uint64_t count, uint32_t kind);
#endif