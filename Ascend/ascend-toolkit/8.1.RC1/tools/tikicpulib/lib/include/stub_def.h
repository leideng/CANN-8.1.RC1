/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file stub_def.h
 * \brief
 */
#ifndef ASCENDC_STUB_DEF_H
#define ASCENDC_STUB_DEF_H
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <thread>
#include <cstdio>
#include <cassert>
#include "kernel_fp16.h"
#include "kernel_bf16.h"
#include "kernel_print_lock.h"

#define __global__
#define __WORKGROUP_LOCAL__
#define __BLOCK_LOCAL__
#define __VEC_SCOPE__
#define __aicore__
#define __gm__
#define __cbuf__
#define __ubuf__
#define __cc__
#define __ca__
#define __cb__
#define __fbuf__
#define __sync_noalias__
#define __sync_alias__
#define __check_sync_alias__
#define __sync_out__
#define __sync_in__
#define __inout_pipe__(...)
#define __in_pipe__(...)
#define __out_pipe__(...)
using vector_u8 = uint8_t[256];
using vector_u16 = uint16_t[128];
using vector_u32 = uint32_t[64];
using vector_s8 = int8_t[256];
using vector_s16 = int16_t[128];
using vector_s32 = int32_t[64];
using vector_s64 = int64_t[32];
using vector_u64 = uint64_t[32];
using vector_bf16 = bfloat16_t[128];
using vector_f16 = half[128];
using vector_f32 = float[64];
using vector_bool = uint8_t; // preg index.
using vector_address = int32_t;

using vector_align = int8_t[32];
#define Mode_Zeroing_Type Literal

#define ARG_STEP 0x1000000000000000

extern int64_t block_idx;
extern int64_t block_num;
extern int64_t g_ubBase;
extern uint64_t g_tilingKey;
extern int32_t g_coreType;
extern std::string g_strCoreType;
extern int32_t g_taskRation;
extern int32_t sub_block_idx;
extern uint64_t* g_workspaceSharedPtr;
extern uint64_t g_fullSizeOfWorkspace;
extern uint64_t g_fixpipeNdNzParam;

enum class KernelMode {
    MIX_MODE = 0,
    AIC_MODE,
    AIV_MODE,
    MIX_AIC_1_1,
};
enum class SocVersion {
    VER_100 = 100,
    VER_200 = 200,
    VER_220 = 220,
    VER_MAX = 0xFFFFFF
};

using ArgInfoT =  struct ArgInfoT {   // parameter info
    std::string argType;  // tensor, tensorlist, workspace, tiling
    std::vector<uint8_t *> addrList;    // addr list
};

extern KernelMode g_kernelMode;
extern SocVersion g_socVersion;
extern std::vector<ArgInfoT> g_argInfoList;
extern std::vector<std::string> g_validArgTypeList;
extern std::vector<std::string> g_tmpFileName;
extern std::vector<int32_t> g_processId;
extern int32_t g_mainPid;
extern int32_t g_processNum;

namespace AscendC {
extern const int MIX_TYPE;
extern const int AIC_TYPE;
extern const int AIV_TYPE;
extern const int PAGE_SIZE;
extern const uint64_t ONE_GIGABYTE;
extern bool g_isVdeq;
constexpr int FLAG_NUM = 16;
constexpr int MAX_CORE_NUM_V220 = 25;
constexpr int MIX_IN_GROUP_CORE_NUM = 3;
constexpr int AIV_IN_GROUP_CORE_NUM = 2;
constexpr uint64_t FFTS_MODE_BITS = 3ull;
constexpr uint64_t FFTS_FLAG_BITS = 15ull;
constexpr uint64_t FFTS_FLAG_CONFIG_BIT_POSITION = 8ull;
constexpr uint64_t FFTS_MODE_CONFIG_BIT_POSITION = 4ull;
constexpr int32_t INTER_BLOCK_MODE = 0;
constexpr int32_t INTER_SUBBLOCK_MODE = 1;
constexpr int32_t INTRA_GROUP_MODE = 2;
constexpr int32_t FFTS_THRESHOLD = 16;
constexpr int32_t FFTS_COUNTER_NUM = 2;

extern uint8_t (*g_syncCounterEachcore)[FLAG_NUM];
extern uint8_t (*g_syncCounterFfts)[FLAG_NUM];
void AddNameArg(const char* name, unsigned long val);
unsigned long GetNameArg(const char* name);
std::string BuildExp(uint64_t val);

inline void InitSocVersion()
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 100)
    g_socVersion = SocVersion::VER_100;
#elif defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
    g_socVersion = SocVersion::VER_200;
#elif defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
    g_socVersion = SocVersion::VER_220;
#else
    g_socVersion = SocVersion::VER_MAX;
#endif
}

bool FileExit(std::string fileName);
std::string GetFileName();
std::string GetCoreName(int idx);
void* GmAlloc(size_t size);
void GmFree(void* ptr);
void CheckGmValied(int argn, uint64_t* argv);
uint64_t GmGetUserSize(uint64_t addr);
void SetGCoreType(int type);
void SetArgInfoList(const std::vector<ArgInfoT> &argInfoList);
void SetKernelMode(KernelMode mode);
void CheckBlockdimForFfts(uint64_t blkdim);
void BacktracePrint(int sig);
typedef struct {
    uint64_t magicCode;
    int fd;
    size_t size;
    char fileName[256];
} ShmMemT;

} // namespace AscendC
#endif // ASCENDC_STUB_DEF_H
