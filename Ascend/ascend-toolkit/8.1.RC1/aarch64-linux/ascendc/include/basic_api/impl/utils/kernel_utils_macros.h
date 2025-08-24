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
 * \file kernel_utils_macros.h
 * \brief
 */
#ifndef ASCENDC_MODULE_UTILS_MACROS_H
#define ASCENDC_MODULE_UTILS_MACROS_H
#define USE_ISA_INS 1
#define GM_ADDR __gm__ uint8_t*

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#include "kernel_macros.h"
#include "kernel_log.h"
#include "kernel_event.h"
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include <set>
#include <map>
#include <sstream>
#include <thread>
#include <iomanip>
#include "stub_def.h"
#include "stub_fun.h"
#endif // ASCENDC_CPU_DEBUG

// this marco is used to define new array with dim
#define ASCENDC_SHAPE(dimValue, ...) \
    dimValue, (const uint32_t[])     \
    {                                \
        __VA_ARGS__                  \
    }

// define macro for deterministic compile options

enum KernelMetaType : uint8_t {
    KERNEL_TYPE_AIV_ONLY,
    KERNEL_TYPE_AIC_ONLY,
    KERNEL_TYPE_MIX_AIV_1_0,
    KERNEL_TYPE_MIX_AIC_1_0,
    KERNEL_TYPE_MIX_AIC_1_1,
    KERNEL_TYPE_MIX_AIC_1_2,
    KERNEL_TYPE_AICORE,
    KERNEL_TYPE_VECTORCORE,
    KERNEL_TYPE_MIX_AICORE,
    KERNEL_TYPE_MIX_VECTOR_CORE,
    KERNEL_TYPE_MAX,
};

enum KernelType {
    K_TYPE_AICORE = 1,              // c100/m200
    K_TYPE_AIC = 2,                 // v220-cube
    K_TYPE_AIV = 3,                 // v220-vec
    K_TYPE_MIX_AIC_MAIN = 4,        // v220 mix cube/vector 1:2
    K_TYPE_MIX_AIV_MAIN = 5,        // v220 mix vector/cube 1:2
    K_TYPE_AIC_ROLLBACK = 6,        // v220-cube，aic rollback
    K_TYPE_AIV_ROLLBACK = 7,        // v220-vec，aiv rollback
    K_TYPE_MAX
};

struct BaseTlv {  // TLV头部定义
    unsigned short type;
    unsigned short len;
};

enum FuncMetaType { // 函数级TLV类型
    F_TYPE_KTYPE = 1, // kernel type tlv
    F_TYPE_CROSS_CORE_SYNC = 2, // cross core sync
    F_TYPE_MIX_TASK_RATION = 3, // MIX CORE TYPE
    F_TYPE_L0_EXCEPTION_DFX = 4, // DFX tlv for header
    F_TYPE_L0_EXCEPTION_DFX_ARGSINFO = 5, // DFX tlv for args info
    F_TYPE_L0_EXCEPTION_DFX_IS_TIK = 6, // DFX tlv mark for TIK
    F_TYPE_MAX
};

enum CrossCoreSyncType { // 函数级TLV类型
    C_TYPE_USE_SYNC = 1, // use cross core sync
    C_TYPE_MAX
};

struct OpSystemRunCfg {
    uint64_t l2Cacheoffset;
};
#ifdef L2_CACHE_HINT
extern __gm__ struct OpSystemRunCfg g_opSystemRunCfg;
#endif // L2_CACHE_HINT


__aicore__ inline void GetCannVersion(__gm__ char*& versionStr, uint64_t& version, uint64_t& timeStamp)
{
#ifdef CANN_VERSION_STR
    versionStr = const_cast<__gm__ char*>(CANN_VERSION_STR);
#else
    versionStr = const_cast<__gm__ char*>("Unknown CANN version");
#endif

#ifdef CANN_TIMESTAMP
    timeStamp = static_cast<uint64_t>(CANN_TIMESTAMP);
#else
    timeStamp = 0;
#endif

#ifdef CANN_VERSION
    version = static_cast<uint64_t>(CANN_VERSION);
#else
    version = 0;
#endif
}


namespace AscendC {
template <typename U>
__aicore__ inline static auto IsLite(int) -> typename U::LiteType;
template <typename U>
__aicore__ inline static auto IsLite(void*) -> U;

template <typename T>
using PrimT = decltype(IsLite<T>(0));

enum class CacheMode {
    CACHE_MODE_DISABLE = 0,
    CACHE_MODE_NORMAL = 1,
    CACHE_MODE_LAST = 2,
    CACHE_MODE_PERSISTENT = 4
};

enum class CacheRwMode {
    READ = 1,
    WRITE = 2,
    RW = 3
};

template<class T, CacheRwMode rwMode = CacheRwMode::RW>
__aicore__ __inline__ __gm__ T* L2CacheAlter(__gm__ T* addr, CacheMode mode)
{
#if defined(L2_CACHE_HINT) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    uint64_t l2CacheOffset = g_opSystemRunCfg.l2Cacheoffset;
    if (mode == CacheMode::CACHE_MODE_DISABLE) {
        return reinterpret_cast<__gm__ T*>(reinterpret_cast<uint64_t>(addr) + l2CacheOffset);
    }
#endif // L2_CACHE_HINT
    return addr;
}
}

struct FunMetaKType {
    BaseTlv head;
    unsigned int ktype;
};

struct FunMetaCrossCoreType {
    BaseTlv head;
    unsigned int usedCrossCoreSync;
};

struct FunMetaMixCoreType {
    BaseTlv head;
    unsigned short taskRation0;
    unsigned short taskRation1;
};

struct FunLevelKType {
    struct FunMetaKType ktypeMeta;
};

struct FunLevelCrossCoreType {
    struct FunMetaKType ktypeMeta;
    struct FunMetaCrossCoreType crossCoreType;
};

struct FunLevelMixCoreType {
    struct FunMetaKType ktypeMeta;
    struct FunMetaMixCoreType mixCoreType;
};

#ifdef __CHECK_FEATURE_AT_PRECOMPILE
#define ENABLE_FEATURE_FOR_COMPILE(f, val) auto __enable_feature_for_compile_##f = val;
#define ENABLE_FEATURE_FOR_TILING(expression, val) auto __enable_custom_tiling val = expression;
#else
#define ENABLE_FEATURE_FOR_COMPILE(f, val)
#define ENABLE_FEATURE_FOR_TILING(expression, val) do { \
    val __verify_tiling_struct;                         \
} while (0)
#endif

#define ENABLE_DETERMINISTIC() ENABLE_FEATURE_FOR_COMPILE(deterministic, 1)
#define KERNEL_TASK_TYPE(key, value)  ENABLE_FEATURE_FOR_COMPILE(key, value)
#define KERNEL_TASK_TYPE_DEFAULT(value)  ENABLE_FEATURE_FOR_COMPILE(default, value)
#define REGISTER_TILING_DEFAULT(tiling_struct)  ENABLE_FEATURE_FOR_TILING(default, tiling_struct)
#define REGISTER_TILING_FOR_TILINGKEY(expression, tiling_struct)  ENABLE_FEATURE_FOR_TILING(expression, tiling_struct)

#define ENABLE_PRINTF() ENABLE_FEATURE_FOR_COMPILE(printf, 1)
#define ENABLE_PRINTF_DUMP_SIZE() ENABLE_FEATURE_FOR_COMPILE(printfBufSize, 1048576)
#define ENABLE_ASSERT() ENABLE_FEATURE_FOR_COMPILE(assert, 1)
#define ENABLE_ASSERT_DUMP_SIZE() ENABLE_FEATURE_FOR_COMPILE(assertBufSize, 1024)

#ifndef ONE_CORE_DUMP_SIZE
    #define ONE_CORE_DUMP_SIZE (1024 * 1024)
#endif

namespace AscendC {
constexpr int32_t MIX = 0;
constexpr int32_t AIC = 1;
constexpr int32_t AIV = 2;
constexpr size_t DUMP_UINTSIZE = ONE_CORE_DUMP_SIZE;
} // namespace AscendC
#if defined(ASCENDC_CPU_DEBUG)
extern int32_t g_coreType;
#define ASCEND_IS_AIV (g_coreType == AscendC::AIV)
#define ASCEND_IS_AIC (g_coreType == AscendC::AIC)
#define ASCEND_IS_NOT_AIV (g_coreType != AscendC::AIV)
#define ASCEND_IS_NOT_AIC (g_coreType != AscendC::AIC)
#else
#if defined(__DAV_C220_CUBE__)
constexpr int32_t g_coreType = AscendC::AIC;
#elif defined(__DAV_C220_VEC__)
constexpr int32_t g_coreType = AscendC::AIV;
#else
constexpr int32_t g_coreType = AscendC::MIX;
#endif
#define ASCEND_IS_AIV constexpr(g_coreType == AscendC::AIV)
#define ASCEND_IS_AIC constexpr(g_coreType == AscendC::AIC)
#define ASCEND_IS_NOT_AIV constexpr(g_coreType != AscendC::AIV)
#define ASCEND_IS_NOT_AIC constexpr(g_coreType != AscendC::AIC)
#endif

#include <stdint.h>
#ifndef TILING_KEY_VAR
#if defined(ASCENDC_CPU_DEBUG)
extern uint64_t g_tilingKey;
#else
#if __CCE_AICORE__ == 200
[[block_local]] uint64_t g_tilingKey;
#else
[[workgroup_local]] __gm__ uint64_t g_tilingKey;
#endif
#endif
#define TILING_KEY_VAR g_tilingKey
#endif

#define TILING_KEY_IS(k) (TILING_KEY_VAR == (k))

#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
constexpr bool g_gm_overflow_check = true;
constexpr uint64_t g_oomAddrRangeMaxSize = 128;
struct OomAddrRange {
    uintptr_t addr[g_oomAddrRangeMaxSize];
    uint64_t len[g_oomAddrRangeMaxSize];
    uint8_t isLevelOnePointer[g_oomAddrRangeMaxSize];
    uint64_t count;
};
__BLOCK_LOCAL__ __inline__ OomAddrRange g_oomAddrArange;
#else
constexpr bool g_gm_overflow_check = false;
#endif

#endif // ASCENDC_MODULE_UTILS_MACROS_H