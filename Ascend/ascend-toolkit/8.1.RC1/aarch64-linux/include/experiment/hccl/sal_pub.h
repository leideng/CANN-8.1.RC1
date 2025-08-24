/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2017-2022. All rights reserved.
 * Description: 系统抽象层公共头文件.
 */

#ifndef HCCL_INC_SAL_PUB_H
#define HCCL_INC_SAL_PUB_H

#include <climits>
#include <chrono>
#include <exception>
#include <securec.h>
#include <string>
#include <map>
#include <vector>
#include <set>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "hccl_ip_address.h"

#ifndef T_DESC
#define T_DESC(_msg, _y) ((_y) ? true : false)
#endif

std::string SalGetEnv(const char *name);

#if T_DESC("库函数封装", true)
constexpr int HCCL_BASE_DECIMAL = 10; // 10进制字符串转换
constexpr int HCCL_BASE_HEX = 16; // 16进制字符串转换

HcclResult  SalStrToInt(const std::string str, int base, s32 &val);
HcclResult  SalStrToULong(const std::string str, int base, u32 &val);
HcclResult  SalStrToULonglong(const std::string str, int base, u64 &val);
HcclResult  SalStrToLonglong(const std::string str, int base, s64 &val);
#endif

#if T_DESC("跨进程处理函数", true)
s32 SalGetPid();
HcclResult SalGetBareTgid(u32 *pid);
u32 SalGetUid();
s32 SalGetTid();
extern HcclResult SalGetUniqueId(char *salUniqueId, int maxLen = INT_MAX);

#endif

#if T_DESC("路径信息函数", true)
HcclResult SalIsDirExist(const std::string &dir, s32 &status);
#endif

#if T_DESC("C字符串处理函数适配", true)
std::string SalTrim(const std::string &s);
#endif

#if T_DESC("设置指定位值函数", true)
void SalSetBitOne(u64 &value, u64 index);
#endif

#if T_DESC("时间处理接口适配", true)
constexpr u32 SOCKET_SLEEP_MILLISECONDS = 1;
constexpr u32 ONE_HUNDRED_MICROSECOND_OF_USLEEP = 100;
constexpr u32 TWO_HUNDRED_MICROSECOND_OF_USLEEP = 200;
constexpr u32 ONE_MILLISECOND_OF_USLEEP = 1000;
constexpr u32 TEN_MILLISECOND_OF_USLEEP = 10000;
constexpr u32 TCP_SEND_THREAD_SLEEP_TWO_HUNDRED_MICROSECOND = 200;
constexpr u32 TIME_S_TO_MS = 1000;
s64 SalGetSysTime();
void SaluSleep(u32 usec);
void SalSleep(u32 sec);
HcclResult SalGetCurrentTimestamp(u64& timestamp);
u64 GetCurAicpuTimestamp();

using HcclUs = std::chrono::steady_clock::time_point;

#define DURATION_US(x) (std::chrono::duration_cast<std::chrono::microseconds>(x))
#define TAKE_TIME_US(x, y) (DURATION_US(x) - DURATION_US(y))
#define TIME_NOW() ({ std::chrono::steady_clock::now(); })
#define CHECK_WARNTIME(x, warntime) do { \
    if ((x) > (warntime))                    \
        HCCL_WARNING("over warning Time\n"); \
} while (0)

#ifdef TIME_PROFILING
#define TIME_PRINT(x) \
    do { \
        auto startTime = TIME_NOW(); \
        auto timeGap = TIME_NOW() - startTime; \
        x; \
        HCCL_ERROR("Time Cost: cost time %llu us %s", TAKE_TIME_US((TIME_NOW() - startTime), (3 * timeGap)), #x); \
    } while (0)
#else
#define TIME_PRINT(x) \
    do { \
        x; \
    } while (0)
#endif
using HcclSystemTime = std::chrono::system_clock::time_point;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define weak_alias(name, aliasname) _weak_alias(name, aliasname)
#define _weak_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((weak, alias(#name)))

#define strong_alias(name, aliasname) _strong_alias(name, aliasname)
#define _strong_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((alias(#name)))

constexpr s32 BUF_SIZE = 1024;
constexpr size_t MEMCPY_THRESHOLD = 1024;
s32 SalLog2(s32 data);
#ifdef __cplusplus
}  // extern "C"
#endif

#if T_DESC("计算类型占用内存大小函数", true)
HcclResult SalGetDataTypeSize(HcclDataType dataType, u32 &dataTypeSize);
#endif

HcclResult GetLocalHostIP(hccl::HcclIpAddress &ip, u32 devPhyid = 0);

HcclResult FindLocalHostIP(std::vector<std::pair<std::string, hccl::HcclIpAddress>> &ifInfos, hccl::HcclIpAddress &ip);
std::string GetLocalServerId(std::string &serverId);
bool IsGeneralServer();
HcclResult IsHostUseDevNic(bool &isHdcMode);
u32 GetNicPort(u32 devicePhyId, const std::vector<u32> &ranksPort, u32 userRank, bool isUseRanksPort);

HcclResult IsAllDigit(const char *strNum);

void SetThreadName(const std::string &threadStr);

#endif  // HCCL_INC_SAL_H
