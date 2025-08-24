//===--------- tunnel_impl.h - CCE Print Header File ---------*- C++-*-===//
//
// Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file defines the two function in device for CCE.
//
//
//===----------------------------------------------------------------------===//
#ifndef CCELIB_PRINT_TUNNEL_H
#define CCELIB_PRINT_TUNNEL_H

#include "payload.h"

namespace cce {
namespace internal {
// debug tunnel struct reside in gm, shared by both host and Kernel.
//
// must make sure the layout is the same in both sides
struct DebugTunnelData {
  PrintPayloadData PrintData;
  DebugTunnelData() {}
};
} // namespace internal
} // namespace cce
#endif