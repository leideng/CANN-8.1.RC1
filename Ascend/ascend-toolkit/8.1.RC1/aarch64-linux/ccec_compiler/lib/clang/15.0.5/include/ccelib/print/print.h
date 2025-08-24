//===-------------------------- print.h -----------------------------------===//
//
// Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file contains the printf in aicore
//
//
//===----------------------------------------------------------------------===//
#ifndef CCELIB_PRINT_PRINT_H
#define CCELIB_PRINT_PRINT_H

#include "printstate.h"

namespace cce {
template <typename T, typename... Ts>
[aicore] void printf(const T *fmt, Ts... Args) {
  if (!fmt) {
    int8_t type = (int8_t)cce::internal::NodeTy::END;
    cce::internal::Write((const char *)&type, 1);
    return;
  }
  cce::internal::PrintState<const T *> s(fmt);
  (s << ... << (Args));
  int8_t type = (int8_t)cce::internal::NodeTy::NORMAL;
  short len = 0;
  int pos = s.GetCurpos();
  while (fmt[pos + len] != '\0') {
    len++;
  }
  if (!len) {
    type = (int8_t)cce::internal::NodeTy::END;
    cce::internal::Write((const char *)&type, 1);
    return;
  }
  cce::internal::Write((const char *)&type, 1);
  cce::internal::Write((const char *)&len, 2);
  cce::internal::Write((fmt + s.GetCurpos()), len);
  char endChar = '\0';
  cce::internal::Write(&endChar, 1);
  type = (int8_t)cce::internal::NodeTy::END;
  cce::internal::Write((const char *)&type, 1);
}
} // namespace cce
#endif
