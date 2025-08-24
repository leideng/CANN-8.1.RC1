//===----------------------------------------------------------------------===//
//
// This file is used for host-side instrumentation of the address sanitizer.
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef __CLANG_CCE_SANITIZER_H__
#define __CLANG_CCE_SANITIZER_H__

// They are define at sanitzer check tool.
extern "C" {
__attribute__((weak)) uint8_t *__sanitizer_init(uint64_t BlockDim);
__attribute__((weak)) void *__sanitizer_finalize(uint8_t *MemInfo,
                                                 uint64_t BlockDim);
}

// They are used for host-sizd  instrumentation.
// insert before kernel launch
inline uint8_t *__Sanitizer_Init(uint32_t BlockNum) {
  if (__sanitizer_init) {
    uint8_t *p = __sanitizer_init(BlockNum);
    return p;
  }
  return reinterpret_cast<uint8_t *>(0x0);
}

// insert after kernel release
inline void __Sanitizer_Finalize(uint8_t *MemInfo, uint32_t BlockNum) {
  if (__sanitizer_finalize) {
    __sanitizer_finalize(MemInfo, BlockNum);
  }
}

#endif // __CLANG_CCE_SANITIZER_H__