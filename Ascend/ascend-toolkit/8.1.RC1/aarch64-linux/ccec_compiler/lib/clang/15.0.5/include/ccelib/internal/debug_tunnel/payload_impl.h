//===--------- payload_impl.h - CCE Print Header File ---------*- C++-*-===//
//
// Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file defines the two function in host for CCE.
//
//
//===----------------------------------------------------------------------===//
#ifndef CCELIB_PRINT_PAYLOAD_IMPL_H
#define CCELIB_PRINT_PAYLOAD_IMPL_H

#include "../../common/common_impl.h"
#include "payload.h"
#include <stdio.h>
#include <string>

namespace cce {
namespace internal {
namespace PrintPayload {
#define VerifyBorder(nextField, maxBuf)                                        \
  if (nextField > maxBuf) {                                                    \
    printf("\nWARNNING: out of bound! try best to print\n");                   \
    return;                                                                    \
  }

void __attribute__((weak)) PrintFormatString(int8_t *&buf, int8_t *maxbuf) {
  // Get format string len
  VerifyBorder((buf + 2), maxbuf);
  short len = *(short *)buf;
  buf += sizeof(len);
  // Get format string
  VerifyBorder((buf + len), maxbuf);
  printf((const char *)buf);
  buf += len;
}

template <typename T>
void __attribute__((weak))
PrintFormatString(int8_t *&buf, int8_t *maxbuf, T param) {
  // Get format string len
  VerifyBorder((buf + 2), maxbuf);
  short len = *(short *)buf;
  buf += sizeof(len);
  // Get format string
  VerifyBorder((buf + len), maxbuf);
  printf((const char *)buf, param);
  buf += len;
}

/*
serializeFloatParam
the serialized structure of the whole printing data contains many records, each
record correspond to one format specifer. the record format is:
[valuetype 1 byte, paramvalue n bytes, formatstr len 2byte, format string m
bytes]. we also append a trailing record containing the last part of the format
string.

for example, 'printf("hello %4.5lld, world", (long long)1);' generate three
records as follows:
Record-1: [INT, 1, 13, "hello %4.5lld"]
Record-2: [NORMAL, 7, ", world"]
Record-3: [END]

1) int, long long, short, unsigned :
[NodeTy = INT 1byte, param value 8 byte,
formatstr len 2byte, format string :"...%+4.5lld"]
2) float :
[NodeTy = FLOAT 1byte, param value 4 byte,
formatstr len 2byte, format string :"...%+4.5llf"]
3) string :
[NodeTy = STRING 1byte, strlen 2 byte, param value n byte,
formatstr len 2byte, format string :"...%+4.5lls"]
4) pointer :
[NodeTy = POINTER 1byte, param value 8 byte,
formatstr len 2byte, format string :"...%+4.5llp"]
5) char :
[NodeTy = POINTER 1byte, param value 1 byte,
formatstr len 2byte, format string :"...%+4.5llc"]
6) normal :
[NodeTy = NORMAL 1byte, formatstr len 2byte, format
string "hello world"]
7) END :
[NodeTy = END 1byte]
*/
void __attribute__((weak))
AnalyzeSerializedData(int8_t *buf, int logSize, int maxSize) {
  int8_t *bufEndAddr = buf + logSize;
  int8_t *maxbuf = buf + maxSize;
  // Get param
  // If the type field exceeds the buffer,
  // the format string field is truncated and the printing stops.
  while (buf < bufEndAddr) {
    VerifyBorder((buf + 1), maxbuf);
    int8_t type = *(int8_t *)buf;
    while (type != cce::internal::END) {
      buf += sizeof(type);
      // Get param value
      switch (type) {
      default:
        break;
      case cce::internal::NodeTy::NORMAL: {
        // [NodeTy = STRING 1byte, strlen 2 byte, param value n byte, formatstr
        // len 2byte, format string :"...%+4.5lls"]
        PrintFormatString(buf, maxbuf);
        break;
      }
      case cce::internal::NodeTy::FLOAT: {
        // [NodeTy = FLOAT 1byte, param value 4 byte, formatstr len 2byte,
        // format string :"...%+4.5llf"]
        VerifyBorder((buf + 4), maxbuf);
        float param = *(float *)buf;
        buf += sizeof(param);
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      // [NodeTy = INT 1byte, param value 8 byte, formatstr len 2byte, format
      // string
      // :"...%+4.5lld"]
      case cce::internal::NodeTy::INT: {
        VerifyBorder((buf + 8), maxbuf);
        long long int param = *(long long int *)buf;
        buf += sizeof(param);
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      case cce::internal::NodeTy::STRING: {
        // [NodeTy = STRING 1byte, strlen 2 byte, param value n byte, formatstr
        // len 2byte, format string :"...%+4.5lls"]
        VerifyBorder((buf + 2), maxbuf);
        short strlen = *(short *)buf;
        buf += sizeof(strlen);
        VerifyBorder((buf + strlen), maxbuf);
        char *param = reinterpret_cast<char *>(buf);
        buf += strlen;
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      case cce::internal::NodeTy::CHAR: {
        // char : [NodeTy = CHAR 1byte, param value 1 byte, formatstr len 2byte,
        // format string :"...%+4.5llc"]
        VerifyBorder((buf + 1), maxbuf);
        char param = *(char *)buf;
        buf += sizeof(param);
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      case cce::internal::NodeTy::POINTER: {
        // [NodeTy = POINTER 1byte, param value 8 byte, formatstr len 2byte,
        // format string :"...%+4.5llc"]
        VerifyBorder((buf + 8), maxbuf);
        void *param = *(void **)buf;
        buf += sizeof(param);
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      }
      VerifyBorder((buf + 1), maxbuf);
      type = *(int8_t *)buf;
    }
    // If end,jump to next type
    buf += 1;
  }
}

void __attribute__((weak))
OnHostInitialize(PrintPayloadData *PrintData, unsigned BlockNum) {
  PrintData->LogBufferSize = BlockMaxSize;
  PrintData->BlockNum = BlockNum;
  int WholeSize =
      (PrintData->LogBufferSize + LogBufferPaddingBytes) * PrintData->BlockNum;
  printf("each block Size is : %d\n", PrintData->LogBufferSize);

  void *Hbm_PrintPayloadData_start_addr = NULL;
  rtError_t error =
      rtMalloc(reinterpret_cast<void **>(&Hbm_PrintPayloadData_start_addr),
               WholeSize, RT_MEMORY_HBM);
  if (error != RT_ERROR_NONE) {
    printf("ERROR:The memory for the printing function on the device side "
           "fails to be allocated.");
    printf("As a result, the printing function fails!\n");
    return;
  }
  PrintData->LogWholeRegion = (__gm__ char *)Hbm_PrintPayloadData_start_addr;
}

void __attribute__((weak))
OnHostFinish(PrintPayloadData *PrintData, void *&Stream) {
  if (!PrintData->LogWholeRegion) {
    return;
  }
  std::size_t WholeSize =
      (PrintData->LogBufferSize + LogBufferPaddingBytes) * PrintData->BlockNum;
  char *hostMemOut2;
  rtError_t error =
      rtMallocHost(reinterpret_cast<void **>(&hostMemOut2), WholeSize);
  if (error != RT_ERROR_NONE) {
    printf("ERROR:The memory for the printing function on the device side "
           "fails to be allocated.");
    printf("As a result, the printing function fails!\n");
    return;
  }
  error = rtMemcpyAsync(hostMemOut2, WholeSize, PrintData->LogWholeRegion,
                        WholeSize, RT_MEMCPY_DEVICE_TO_HOST, Stream);
  if (error != RT_ERROR_NONE) {
    printf("ERROR:The memory copy of the device print on fails,");
    printf("and the printing function is invalid!\n");
    return;
  }
  error = rtStreamSynchronize(Stream);
  if (error != RT_ERROR_NONE) {
    printf("ERROR:Synchronous waiting for the device print failed.");
    printf("The printing function is invalid!\n");
    return;
  }
  char *outRaw2 = static_cast<char *>(hostMemOut2);
  const char *Line = "-------------------------------------------------------"
                     "----------------------";
  printf("%s\n", Line);
  printf("---------------------------------HiIPU "
         "Print---------------------------------\n");
  printf("%s\n", Line);
  for (int B = 0; B < PrintData->BlockNum; B++) {
    printf("==> Block %d\n", B);
    char *Log =
        (outRaw2 + (PrintData->LogBufferSize + LogBufferPaddingBytes) * B);
    size_t LogSize = *reinterpret_cast<size_t *>(Log);
    if (LogSize > PrintData->LogBufferSize || LogSize < 0) {
      printf(" LOG SIZE ERROR !!! \n");
      printf(" log size needed =  %d ", LogSize);
      printf(" , buf size =   %d\n", PrintData->LogBufferSize);
      LogSize = PrintData->LogBufferSize;
    }
    int8_t *Buf =
        reinterpret_cast<int8_t *>(Log + LogBufferPaddingBytes); // data addr
    printf(" Data:  \n");
    AnalyzeSerializedData(Buf, LogSize, PrintData->LogBufferSize);
    printf("\n");
    printf("%s\n", Line);
  }
  error = rtFree(PrintData->LogWholeRegion);
  if (error != RT_ERROR_NONE) {
    printf("ERROR:The memory free of the device print fails\n");
    return;
  }
  error = rtFreeHost(hostMemOut2);
  if (error != RT_ERROR_NONE) {
    printf("ERROR:The memory free of the device print fails\n");
    return;
  }
}

[aicore] void __attribute__((weak))
OnKernelInitialize(__gm__ PrintPayloadData *PrintData) {
  if (!PrintData->LogWholeRegion) {
    return;
  }
  __gm__ char *LogBuffer =
      PrintData->LogWholeRegion +
      (PrintData->LogBufferSize + LogBufferPaddingBytes) * get_block_idx();
  *(__gm__ size_t *)(LogBuffer) = 0; // initialize log size to 0
}

[aicore] void __attribute__((weak))
OnKernelFinish(__gm__ PrintPayloadData *PrintData) {
  if (!PrintData->LogWholeRegion) {
    return;
  }
  dcci(nullptr, 1);
}
} // namespace PrintPayload
} // namespace internal
} // namespace cce
#endif
