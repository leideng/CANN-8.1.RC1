//===--------- __fbalib_aicore.h - CCE Print Header File ---------*- C++-*-===//
//
// Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for CCE.
//
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CCELIB_AICORE_H
#define CLANG_CCELIB_AICORE_H

// We separate the aicore only header files from aicore and host mixed files.
// This can avoid mangling errors in host side compiling when generating aicore
// function AST.
#include "print/print.h"

#endif