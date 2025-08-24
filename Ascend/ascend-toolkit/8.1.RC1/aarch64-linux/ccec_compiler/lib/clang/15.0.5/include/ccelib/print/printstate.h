//===-------------------------- format.h ----------------------------------===//
//
// Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of PrintState
//
//
//===----------------------------------------------------------------------===//

#ifndef CCELIB_PRINT_PRINTSTATE_H
#define CCELIB_PRINT_PRINTSTATE_H

#include "../common/common_impl.h"
#include <type_traits>

namespace cce {
namespace internal {
template <typename FromTy, typename ToTy, typename = void> struct ConvertTo {
  [aicore] __attribute__((always_inline)) static ToTy convert(FromTy input) {
    return ToTy();
  }
};

template <typename FromTy, typename ToTy>
struct ConvertTo<
    FromTy, ToTy,
    typename std::enable_if<std::is_convertible<FromTy, ToTy>::value &&
                            !std::is_same<ToTy, float>::value>::type> {
  [aicore] __attribute__((always_inline)) static ToTy convert(FromTy input) {
    return static_cast<ToTy>(input);
  }
};

template <typename FromTy, typename ToTy>
struct ConvertTo<
    FromTy, ToTy,
    typename std::enable_if<std::is_same<ToTy, float>::value &&
                            std::is_same<FromTy, float>::value>::type> {
  [aicore] __attribute__((always_inline)) static ToTy convert(FromTy input) {
    return input;
  }
};

template <typename FromTy, typename ToTy>
struct ConvertTo<
    FromTy, ToTy,
    typename std::enable_if<std::is_same<ToTy, float>::value &&
                            !std::is_same<FromTy, float>::value>::type> {
  [aicore] __attribute__((always_inline)) static ToTy convert(FromTy input) {
    return 0.0f;
  }
};

template <typename FromTy>
[aicore] __attribute__((always_inline)) long long int
ConvertToPtr(FromTy *ptr) {
  return *(long long int *)&ptr;
}

template <typename FromTy>
[aicore] __attribute__((always_inline)) long long int ConvertToPtr(FromTy val) {
  return 0LL;
}

template <typename T>
[aicore] void Write(T const *str, size_t size) {
  __gm__ DebugTunnelData *DTData = DebugTunnel::GetKernelInstance();
  if (!DTData) {
    return;
  }
  if (!DTData->PrintData.LogWholeRegion) {
    return;
  }
  __gm__ char *LogBuffer =
      DTData->PrintData.LogWholeRegion +
      (DTData->PrintData.LogBufferSize + LogBufferPaddingBytes) *
          get_block_idx();
  __gm__ size_t *pLogSize = (__gm__ size_t *)(LogBuffer);

  int32_t bufMaxSize = DTData->PrintData.LogBufferSize;
  int32_t bufCurSize = *pLogSize;
  int32_t bufLeftSize = bufMaxSize - bufCurSize;
  __gm__ char *buf = LogBuffer + LogBufferPaddingBytes + bufCurSize;

  size_t strLen = size;
  // not overflow
  if (bufLeftSize <= 0) {
    return;
  }
  // if print size is larger than the max buffer size, we record the print size
  // in *PLogSize, and only print the max size characters.
  if (strLen > bufLeftSize) {
    strLen = bufLeftSize;
  }
  while (strLen--) {
    *buf++ = *str++;
  }
  bufLeftSize -= size;

  *pLogSize = bufMaxSize - bufLeftSize;
}

template <typename Addr>
class PrintState {
public:
  [aicore] PrintState(const __gm__ char *str) {
    fmt = str;
    curpos = 0;
  }

      [aicore] PrintState(const char *str) {
    fmt = str;
    curpos = 0;
  }
  [aicore] __attribute__((always_inline)) int GetCurpos() { return curpos; }

  [aicore] __attribute__((always_inline)) bool isdigit(char ch) {
    return (ch >= '0') && (ch <= '9');
  }
  [aicore] __attribute__((always_inline)) void ParseFlag();

  [aicore] __attribute__((always_inline)) void ParseWidth();

  [aicore] __attribute__((always_inline)) void ParseLength();

  [aicore] __attribute__((always_inline)) void ParsePrec();

  [aicore] __attribute__((always_inline)) void
  WriteFormatString(int bufferstar);

  template <typename T>[aicore] PrintState &operator<<(T arg);

private:
  int curpos;
  Addr fmt;
};

template <typename T>
[aicore] __attribute__((always_inline)) void PrintState<T>::ParseFlag() {
  while (true) {
    switch (fmt[curpos]) {
    case '0':
    case '-':
    case '+':
    case ' ':
    case '#':
      curpos++;
      break;
    default:
      return;
    }
  }
}

template <typename T>
[aicore] __attribute__((always_inline)) void PrintState<T>::ParseLength() {
  switch (fmt[curpos]) {
  case 'l':
  case 'z':
  case 'h':
    curpos++;
    break;
  default:
    break;
  }
  if (fmt[curpos] == 'l') {
    curpos++;
  }
}

template <typename T>
[aicore] __attribute__((always_inline)) void PrintState<T>::ParseWidth() {
  while (isdigit(fmt[curpos])) {
    curpos++;
  }
}

template <typename T>
[aicore] __attribute__((always_inline)) void PrintState<T>::ParsePrec() {
  if (fmt[curpos] == '.') {
    curpos++;
    while (isdigit(fmt[curpos])) {
      curpos++;
    }
  }
}

template <typename T>
[aicore] __attribute__((always_inline)) void
PrintState<T>::WriteFormatString(int bufferstart) {
  short strlen = curpos - bufferstart + 1;
  Write((const char *)&strlen, 2);
  Write(&fmt[bufferstart], strlen - 1);
  char endChar = '\0';
  Write(&endChar, 1);
}

template <typename Addr>
template <typename T>
[aicore] PrintState<Addr> &PrintState<Addr>::operator<<(T arg) {
  int bufferstart = curpos;
Start:
  while (fmt[curpos] != '%' && fmt[curpos] != '\0') {
    curpos++;
  }

  // has param but not %..
  if (fmt[curpos] == '\0') {
    int8_t type = (int8_t)cce::internal::NodeTy::NORMAL;
    Write((const char *)&type, 1);
    WriteFormatString(bufferstart);
    return *this;
  }
  curpos++;
  ParseFlag();
  ParseWidth();
  ParsePrec();
  ParseLength();
  int8_t type = (int8_t)cce::internal::NodeTy::NORMAL;
  // serializeFloatParam
  // serialized structure is:
  // [type 1 byte,param n bytes,formatstring m bytes]....
  // int, long long, short, unsigned :
  // [NodeTy = INT 1byte, param value 8 byte,
  // formatlen 2byte, format string :"...%+4.5lld"]
  // float :
  // [NodeTy = FLOAT 1byte, param value 4 byte,
  // formatlen 2byte, format string :"...%+4.5llf"]
  // string :
  // [NodeTy = STRING 1byte, strlen 2 byte, param value n byte,
  // formatlen 2byte, format string :"...%+4.5lls"]
  // pointer :
  // [NodeTy = POINTER 1byte, param value 8 byte,
  // formatlen 2byte, format string :"...%+4.5llp"]
  // char :
  // [NodeTy = CHAR 1byte, param value 1 byte,
  // formatlen 2byte, format string :"...%+4.5llc"]
  // normal :
  // [NodeTy = NORMAL 1byte, formatlen 2byte,
  // format string "hello world"]
  switch (fmt[curpos]) {
  default:
    goto Start;
  case 's': {
    curpos++;
    const __gm__ char *verifiedType =
        ConvertTo<T, const __gm__ char *>::convert(arg);
    type = (int8_t)cce::internal::NodeTy::STRING;
    Write((const char *)&type, 1);
    if (!verifiedType) {
      char zero = '\0';
      Write(&zero, 1);
    }
    short len = 0;
    while (verifiedType[len] != '\0') {
      len++;
    }
    Write((const char *)&len, 2);
    Write(verifiedType, len);
    WriteFormatString(bufferstart);
    break;
  }
  case 'd':
  case 'i':
  case 'x':
  case 'X':
  case 'o':
  case 'u': {
    curpos++;
    long long int verifiedType = ConvertTo<T, long long int>::convert(arg);
    type = (int8_t)cce::internal::NodeTy::INT;
    Write((const char *)&type, 1);
    Write((const char *)&verifiedType, 8);
    WriteFormatString(bufferstart);
    break;
  }
  case 'f': {
    curpos++;
    float verifiedType = ConvertTo<T, float>::convert(arg);
    type = (int8_t)(cce::internal::NodeTy::FLOAT);
    Write((const char *)&type, 1);
    Write((const char *)&verifiedType, 4);
    WriteFormatString(bufferstart);
    break;
  }
  case 'p': {
    curpos++;
    long long int verifiedType = ConvertToPtr(arg);
    type = (int8_t)(cce::internal::NodeTy::POINTER);
    Write((const char *)&type, 1);
    Write((const char *)&verifiedType, 8);
    WriteFormatString(bufferstart);
    break;
  }
  case 'c': {
    curpos++;
    char verifiedType = ConvertTo<T, char>::convert(arg);
    type = (int8_t)(cce::internal::NodeTy::CHAR);
    Write((const char *)&type, 1);
    Write(&verifiedType, 1);
    WriteFormatString(bufferstart);
    break;
  }
  case '%': {
    curpos++;
    goto Start;
  }
  }
  return *this;
}

} // namespace internal
} // namespace cce

#endif