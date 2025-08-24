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
 * \file kernel_log.h
 * \brief
 */
#ifndef ASCENDC_MODULE_KERNEL_LOG_INTF_H
#define ASCENDC_MODULE_KERNEL_LOG_INTF_H

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include <string>
#include <map>
#include <csignal>
#include <cstdio>
#include <unistd.h>
#include "stub_def.h"

namespace AscendC {
#define ASCENDC_ASSERT(cond, behavior) \
    do {                               \
        if (!(cond)) {                 \
            behavior;                  \
            raise(SIGABRT);            \
        }                              \
    } while (0)

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
#define ASCENDC_REPORT_OVERFLOW_MEM(cond)                                                \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            raise(SIGABRT);                                                              \
        }                                                                                \
    } while (0)
#else
#define ASCENDC_REPORT_OVERFLOW_MEM(cond)
#endif

// Check xxx api is not supported in cpu
#define ASCENDC_REPORT_NOT_SUPPORT(cond, apiMsg)                                         \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            KERNEL_LOG(KERNEL_ERROR, "%s is not supported on current device.", apiMsg);  \
            raise(SIGABRT);                                                              \
        }                                                                                \
    } while (0)

// Check value in range [low, high]
#define ASCENDC_CHECK_VALUE_RANGE(value, rangeLow, rangeHigh, paramName, apiMsg)                       \
    do {                                                                                               \
        if (value < rangeLow || value > rangeHigh) {                                                   \
            KERNEL_LOG(KERNEL_ERROR, "Failed to check %s value in %s, its valid range is "             \
                "%s ~ %s, current value is %s.", paramName, apiMsg, std::to_string(rangeLow).c_str(),  \
                std::to_string(rangeHigh).c_str(), std::to_string(value).c_str());                     \
            raise(SIGABRT);                                                                            \
        }                                                                                              \
    } while (0)

#define ASCENDC_CHECK_TENSOR_PTR_ALIGN(tensorPtr, tPos, alignBytes, tensorName, apiMsg)                \
    do {                                                                                               \
        uint64_t tensorAddr = TransUBAddr<tPos>(reinterpret_cast<uint64_t>(tensorPtr));                \
        if (tensorAddr % alignBytes != 0) {                                                            \
            KERNEL_LOG(KERNEL_ERROR, "Failed to check %s start address alignment in %s, "              \
                "its start address must align with %dB.", tensorName, apiMsg, alignBytes);             \
            raise(SIGABRT);                                                                            \
        }                                                                                              \
    } while (0)

// Check tensor tposition with condition judgement
#define ASCENDC_CHECK_TPOSITION(cond, tensorName, tposName, apiMsg, curPos)                            \
    do {                                                                                               \
        if (!(cond)) {                                                                                 \
            KERNEL_LOG(KERNEL_ERROR, "Failed to check %s tensor position in %s, supported positions "  \
                "are %s, current position is %s.", tensorName, apiMsg, tposName, curPos.c_str());      \
            raise(SIGABRT);                                                                            \
        }                                                                                              \
    } while (0)

// Report error when failed cpu check
#define ASCENDC_REPORT_CHECK_ERROR(apiMsg, funcType)                                        \
    do {                                                                                    \
        if (funcType == KernelFuncType::MASK_COUNT_MODE) {                                  \
            KERNEL_LOG(KERNEL_ERROR, "Failed to pass %s mask count mode check.", apiMsg);   \
        } else if (funcType == KernelFuncType::MASK_BIT_MODE) {                             \
            KERNEL_LOG(KERNEL_ERROR, "Failed to pass %s mask bit mode check.", apiMsg);     \
        } else if (funcType == KernelFuncType::CALCOUNT_MODE) {                             \
            KERNEL_LOG(KERNEL_ERROR, "Failed to pass %s calcount mode check.", apiMsg);     \
        } else {                                                                            \
            KERNEL_LOG(KERNEL_ERROR, "Failed to pass %s check.", apiMsg);                   \
        }                                                                                   \
        raise(SIGABRT);                                                                     \
    } while (0)

enum KernelFuncType : uint8_t {
    NONE_MODE,
    MASK_COUNT_MODE,    // mask
    MASK_BIT_MODE,      // mask[]
    CALCOUNT_MODE       // calcount
};

enum class LogLevel : uint8_t {
    KERNEL_DEBUG = 0,
    KERNEL_INFO = 1,
    KERNEL_WARN = 2,
    KERNEL_ERROR = 3,
};
}  // namespace AscendC

#define KERNEL_LOG(level, format, ...) KERNEL_LOG_##level(format, ##__VA_ARGS__)

#if __CCE_AICORE__ == 220

namespace AscendC {
inline std::string GenCoreTypeStr()
{
    std::string coreTypeStr = "";
    if (g_coreType == AscendC::AIC_TYPE) {
        coreTypeStr = "AIC_";
    } else if (g_coreType == AscendC::AIV_TYPE) {
        coreTypeStr = "AIV_";
    } else {
        coreTypeStr = "MIX_";
    }
    coreTypeStr += std::to_string(sub_block_idx);
    return coreTypeStr;
}

inline std::string GenBlockStr()
{
    std::string blockStr = "Block_";
    blockStr += std::to_string(block_idx);
    return blockStr;
}
} // namespace AscendC

#define KERNEL_LOG_KERNEL_DEBUG(format, ...)                                                                   \
    do {                                                                                                       \
        std::string coreTypeStr = AscendC::GenCoreTypeStr();                                                   \
        std::string blockStr = AscendC::GenBlockStr();                                                         \
        printf("[DEBUG][%s][%s][%s:%u][%s][%u] " format "\n", blockStr.c_str(), coreTypeStr.c_str(), __FILE__, \
            __LINE__, __FUNCTION__, (uint32_t)getpid(), ##__VA_ARGS__);                                        \
    } while (0)

#define KERNEL_LOG_KERNEL_INFO(format, ...)                                                                   \
    do {                                                                                                      \
        std::string coreTypeStr = AscendC::GenCoreTypeStr();                                                  \
        std::string blockStr = AscendC::GenBlockStr();                                                        \
        printf("[INFO][%s][%s][%s:%u][%s][%u] " format "\n", blockStr.c_str(), coreTypeStr.c_str(), __FILE__, \
            __LINE__, __FUNCTION__, (uint32_t)getpid(), ##__VA_ARGS__);                                       \
    } while (0)

#define KERNEL_LOG_KERNEL_WARN(format, ...)                                                                   \
    do {                                                                                                      \
        std::string coreTypeStr = AscendC::GenCoreTypeStr();                                                  \
        std::string blockStr = AscendC::GenBlockStr();                                                        \
        printf("[WARN][%s][%s][%s:%u][%s][%u] " format "\n", blockStr.c_str(), coreTypeStr.c_str(), __FILE__, \
            __LINE__, __FUNCTION__, (uint32_t)getpid(), ##__VA_ARGS__);                                       \
    } while (0)

#define KERNEL_LOG_KERNEL_ERROR(format, ...)                                                                   \
    do {                                                                                                       \
        std::string coreTypeStr = AscendC::GenCoreTypeStr();                                                   \
        std::string blockStr = AscendC::GenBlockStr();                                                         \
        printf("[ERROR][%s][%s][%s:%u][%s][%u] " format "\n", blockStr.c_str(), coreTypeStr.c_str(), __FILE__, \
            __LINE__, __FUNCTION__, (uint32_t)getpid(), ##__VA_ARGS__);                                        \
    } while (0)

#else

#define KERNEL_LOG_KERNEL_DEBUG(format, ...)                                                                  \
    do {                                                                                                      \
        std::string blockStr = "Core_";                                                                       \
        blockStr += std::to_string(block_idx);                                                                \
        printf("[DEBUG][%s][%s:%u][%s][%u] " format "\n", blockStr.c_str(), __FILE__, __LINE__, __FUNCTION__, \
            (uint32_t)getpid(), ##__VA_ARGS__);                                                               \
    } while (0)

#define KERNEL_LOG_KERNEL_INFO(format, ...)                                                                  \
    do {                                                                                                     \
        std::string blockStr = "Core_";                                                                      \
        blockStr += std::to_string(block_idx);                                                               \
        printf("[INFO][%s][%s:%u][%s][%u] " format "\n", blockStr.c_str(), __FILE__, __LINE__, __FUNCTION__, \
            (uint32_t)getpid(), ##__VA_ARGS__);                                                              \
    } while (0)

#define KERNEL_LOG_KERNEL_WARN(format, ...)                                                                  \
    do {                                                                                                     \
        std::string blockStr = "Core_";                                                                      \
        blockStr += std::to_string(block_idx);                                                               \
        printf("[WARN][%s][%s:%u][%s][%u] " format "\n", blockStr.c_str(), __FILE__, __LINE__, __FUNCTION__, \
            (uint32_t)getpid(), ##__VA_ARGS__);                                                              \
    } while (0)

#define KERNEL_LOG_KERNEL_ERROR(format, ...)                                                                  \
    do {                                                                                                      \
        std::string blockStr = "Core_";                                                                       \
        blockStr += std::to_string(block_idx);                                                                \
        printf("[ERROR][%s][%s:%u][%s][%u] " format "\n", blockStr.c_str(), __FILE__, __LINE__, __FUNCTION__, \
            (uint32_t)getpid(), ##__VA_ARGS__);                                                               \
    } while (0)

#endif

#else

#define KERNEL_LOG(level, format, ...)
#define ASCENDC_ASSERT(cond, behavior)
#define ASCENDC_REPORT_NOT_SUPPORT(cond, apiMsg)
#define ASCENDC_CHECK_VALUE_RANGE(value, rangeLow, rangeHigh, paramName, apiMsg)
#define ASCENDC_CHECK_TENSOR_PTR_ALIGN(tensorPtr, tPos, alignBytes, tensorName, apiMsg)
#define ASCENDC_CHECK_TPOSITION(cond, tensorName, tposName, apiMsg, curPos)
#define ASCENDC_REPORT_CHECK_ERROR(apiMsg, funcType)
#define ASCENDC_REPORT_OVERFLOW_MEM(cond)

#endif

namespace AscendC {
template <class... Args>
__aicore__ inline void AssertImpl(__gm__ const char* fmt, Args&&... args);
// assert define
#ifdef ASCENDC_CPU_DEBUG
#define ASSERT_MSG(expr, fmt, ...)                                                                             \
    do {                                                                                                       \
        if (!(expr)) {                                                                                         \
            fprintf(stderr, "[ASSERT] %s:%u: Assertion `%s' " fmt, __FILE__, __LINE__, #expr, ## __VA_ARGS__); \
            abort();                                                                                           \
        }                                                                                                      \
    } while (0)
#else

#if defined(CANN_VERSION_STR) && defined(CANN_TIMESTAMP)
#define ASSERT_MSG(expr, fmt, ...)  \
    do {                                                                                                   \
        ENABLE_ASSERT();                                                                                   \
        ENABLE_ASSERT_DUMP_SIZE();                                                                         \
        if (!(expr)) {                                                                                     \
            AscendC::AssertImpl("[ASSERT] [CANN_VERSION : %s][TimeStamp : %u] %s:%u: Assertion `%s' " fmt, \
            (__gm__ const char*)(CANN_VERSION_STR), static_cast<uint64_t>(CANN_TIMESTAMP),                 \
            __FILE__, __LINE__, #expr, ##__VA_ARGS__);                                                     \
            trap();                                                                                        \
        }                                                                                                  \
    } while (0)
#else
#define ASSERT_MSG(expr, fmt, ...)                                                                                \
    do {                                                                                                          \
        ENABLE_ASSERT();                                                                                          \
        ENABLE_ASSERT_DUMP_SIZE();                                                                                \
        if (!(expr)) {                                                                                            \
            AscendC::AssertImpl("[ASSERT] %s:%u: Assertion `%s' " fmt, __FILE__, __LINE__, #expr, ##__VA_ARGS__); \
            trap();                                                                                               \
        }                                                                                                         \
    } while (0)
#endif
#endif

#ifdef ASCENDC_DUMP
#define VA_ARGS_IS_EMPTY(...) (sizeof(#__VA_ARGS__) == 1)

#define ASCENDC_DEBUG_ASSERT_IMPL(expr, ...)      \
    do {                                          \
        if (VA_ARGS_IS_EMPTY(__VA_ARGS__)) {      \
            ASSERT_MSG(expr, "\n");               \
        } else {                                  \
            ASSERT_MSG(expr, __VA_ARGS__);        \
        }                                         \
    } while (0)
#else
#define ASCENDC_DEBUG_ASSERT_IMPL(...)
#endif

#ifdef ASCENDC_DEBUG
#define ASCENDC_DEBUG_ASSERT(...) ASCENDC_DEBUG_ASSERT_IMPL(__VA_ARGS__)
#else
#define ASCENDC_DEBUG_ASSERT(...)
#endif
}

#endif // ASCENDC_MODULE_KERNEL_LOG_INTF_H