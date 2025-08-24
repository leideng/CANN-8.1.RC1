//===--------- common_impl.h - CCE Print Header File ---------*- C++-*-===//
//
// Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file defines the NodeTy and buffer size for device print.
//
//
//===----------------------------------------------------------------------===//
#ifndef CCELIB_PRINT_COMMON_IMPL_H
#define CCELIB_PRINT_COMMON_IMPL_H

#include "runtime.h"

namespace cce {
namespace internal {
enum NodeTy { END, NORMAL, FLOAT, INT, CHAR, STRING, POINTER };
} // namespace internal
} // namespace cce
#endif
