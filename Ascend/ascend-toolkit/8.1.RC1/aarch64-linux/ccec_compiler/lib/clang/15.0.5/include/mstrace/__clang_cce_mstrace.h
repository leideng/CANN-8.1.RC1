//===----------------------------------------------------------------------===//
//
// This file is used for host-side instrumentation of the address sanitizer.
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef __CLANG_CCE_MSTRACE_H__
#define __CLANG_CCE_MSTRACE_H__

// They are define at trace check tool.
extern "C" {
__attribute__((weak)) uint8_t *__mstrace_init(uint64_t BlockDim);
__attribute__((weak)) void *__mstrace_finalize(uint8_t *MemInfo,
                                               uint64_t BlockDim);
}

// They are used for host-sizd  instrumentation.
// insert before kernel launch
inline uint8_t *__MSTrace_Init(uint32_t BlockNum) {
  if (__mstrace_init) {
    uint8_t *p = __mstrace_init(BlockNum);
    return p;
  }
  return reinterpret_cast<uint8_t *>(0x0);
}

// insert after kernel release
inline void __MSTrace_Finalize(uint8_t *MemInfo, uint32_t BlockNum) {
  if (__mstrace_finalize) {
    __mstrace_finalize(MemInfo, BlockNum);
  }
}

#endif // __CLANG_CCE_MSTRACE_H__