//===--------- payload.h - CCE Print Header File ---------*- C++-*-===//
//
// Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PrintPayloadData for CCE.
//
//
//===----------------------------------------------------------------------===//
#ifndef CCELIB_PRINT_PAYLOAD_H
#define CCELIB_PRINT_PAYLOAD_H

/*
  we choose 64 here because the largest cacheline size is 64.
*/
#define LogBufferPaddingBytes 64
#define BlockMaxSize 16 * 1024

namespace cce {
namespace internal {
struct PrintPayloadData {
  /*
  LogWholeRegion is a data buffer for the whole kernel configured with block
  number BlockNum. the size of the LogWholeRegion is (LogBufferSize +
  LogBufferPaddingBytes) * BlockNum.

  Each kernel block has a seperate region inside the LogwholeRegin, occupying
  [ (LogBufferSize + LogBufferPaddingBytes) * BlockIndex, (LogBufferSize +
  LogBufferPaddingBytes) * (BlockIndex + 1) ). Each block region has a padding
  in the end, whih the size of LogBufferPaddingBytes.
  */
  __gm__ char *LogWholeRegion;
  unsigned BlockNum;
  size_t LogBufferSize;
  PrintPayloadData()
      : LogWholeRegion((__gm__ char *)nullptr), LogBufferSize(0), BlockNum(0) {}
};
} // namespace internal
} // namespace cce
#endif
