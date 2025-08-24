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
 * \file kern_fwk.h
 * \brief
 */
#ifndef ASCENDC_ICPU_FWK_H
#define ASCENDC_ICPU_FWK_H
#include <sys/stat.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <unordered_map>
#include <sys/wait.h>
#include "stub_def.h"
#include "securec.h"
#include "kernel_common.h"

#define va_args_num(...) VA_ARG_NX_(__VA_ARGS__, VA_ARG_NUM())
#define VA_ARG_NX_(...) VA_ARGN_(__VA_ARGS__)
#define VA_ARGN_(a32, a31, a30, a29, a28, a27, a26, a25, a24, a23, a22, a21, a20, a19, a18, a17, a16, a15, a14, a13, \
    a12, a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, N, ...)                                                       \
    N
#define VA_ARG_NUM()                                                                                                 \
    32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, \
        2, 1, 0

#define VA_SET(arr, i, n, val)                                    \
    do {                                                          \
        auto val_ = val;                                          \
        (i < n) ? (arr)[i] = (uint64_t)(*(uint64_t*)(&val_)) : 0; \
    } while (0)

#define VA_VAL_SET(arr, n, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, \
    a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, ...)                                                 \
    do {                                                                                                             \
        VA_SET(arr, 0, n, a0);                                                                                       \
        VA_SET(arr, 1, n, a1);                                                                                       \
        VA_SET(arr, 2, n, a2);                                                                                       \
        VA_SET(arr, 3, n, a3);                                                                                       \
        VA_SET(arr, 4, n, a4);                                                                                       \
        VA_SET(arr, 5, n, a5);                                                                                       \
        VA_SET(arr, 6, n, a6);                                                                                       \
        VA_SET(arr, 7, n, a7);                                                                                       \
        VA_SET(arr, 8, n, a8);                                                                                       \
        VA_SET(arr, 9, n, a9);                                                                                       \
        VA_SET(arr, 10, n, a10);                                                                                     \
        VA_SET(arr, 11, n, a11);                                                                                     \
        VA_SET(arr, 12, n, a12);                                                                                     \
        VA_SET(arr, 13, n, a13);                                                                                     \
        VA_SET(arr, 14, n, a14);                                                                                     \
        VA_SET(arr, 15, n, a15);                                                                                     \
        VA_SET(arr, 16, n, a16);                                                                                     \
        VA_SET(arr, 17, n, a17);                                                                                     \
        VA_SET(arr, 18, n, a18);                                                                                     \
        VA_SET(arr, 19, n, a19);                                                                                     \
        VA_SET(arr, 20, n, a20);                                                                                     \
        VA_SET(arr, 21, n, a21);                                                                                     \
        VA_SET(arr, 22, n, a22);                                                                                     \
        VA_SET(arr, 23, n, a23);                                                                                     \
        VA_SET(arr, 24, n, a24);                                                                                     \
        VA_SET(arr, 25, n, a25);                                                                                     \
        VA_SET(arr, 26, n, a26);                                                                                     \
        VA_SET(arr, 27, n, a27);                                                                                     \
        VA_SET(arr, 28, n, a28);                                                                                     \
        VA_SET(arr, 29, n, a29);                                                                                     \
        VA_SET(arr, 30, n, a30);                                                                                     \
        VA_SET(arr, 31, n, a31);                                                                                     \
    } while (0)

#define va_args_get(arr, n, ...)                                                                                     \
    VA_VAL_SET(arr, n, ##__VA_ARGS__, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  \
        0, 0, 0, 0, 0, 0)

inline void CheckSysWorspaceSet(uint8_t* ptr)
{
    if (ptr == nullptr) {
        printf("[error]g_sysWorkspaceReserved is null, g_sysWorkspaceReserved has been set or not\n");
    }
}

inline void AscendCExit()
{
    exit(1);
}

inline void Handler(int sig)
{
    if (sig != SIGINT) {
        AscendC::BacktracePrint(sig);
    }
    AscendC::KernelPrintLock::FreeLock();
    AscendC::ProcessLock::FreeLock();
    if (getpid() == g_mainPid) {
        for (int32_t idx = 0; idx < g_processNum; idx++) {
            int status = 0;
            waitpid(g_processId[idx], &status, 0);
            std::cout << "[pid " + std::to_string(g_processId[idx]) + \
                "] exit status:" + std::to_string(status) << std::endl;
        }
        for (auto& tmpFile : g_tmpFileName) {
            struct stat buffer;
            if (stat(tmpFile.c_str(), &buffer) == 0) {
                remove(tmpFile.c_str());
            }
        }
    }
    AscendCExit();
}

#define ICPU_SET_TILING_KEY(tilingKey) \
    do {                               \
        g_tilingKey = tilingKey;       \
    } while (0)

#ifdef ASCENDC_NPUCHK_OFF
#define ICPU_RUN_KF(func, blkdim, ...)                                                                   \
    do {                                                                                                 \
        g_mainPid = getpid();                                                                            \
        AscendC::CheckBlockdimForFfts(blkdim);                                                           \
        AscendC::InitSocVersion();                                                                       \
        uint64_t kargs[32];                                                                              \
        constexpr size_t workspaceSize = AscendC::RESERVED_WORKSPACE;                                    \
        uint8_t* sysWorkSpacePtr = (uint8_t*)AscendC::GmAlloc(workspaceSize);                            \
        memset_s(sysWorkSpacePtr, workspaceSize, 0, workspaceSize);                                      \
        constexpr size_t fftsCounterSize = AscendC::MAX_CORE_NUM_V220 * AscendC::MIX_IN_GROUP_CORE_NUM * \
            AscendC::FLAG_NUM * AscendC::FFTS_COUNTER_NUM;                                               \
        void* ffts_addr = AscendC::GmAlloc(fftsCounterSize);                                             \
        memset_s(ffts_addr, fftsCounterSize, 0, fftsCounterSize);                                        \
        CheckSysWorspaceSet(sysWorkSpacePtr);                                                            \
        g_sysWorkspaceReserved = sysWorkSpacePtr;                                                        \
        int32_t argn = va_args_num(__VA_ARGS__);                                                         \
        va_args_get(kargs, argn, ##__VA_ARGS__);                                                         \
        AscendC::StubInit();                                                                             \
        block_num = blkdim;                                                                              \
        int processNum = get_process_num();                                                              \
        g_processNum = processNum;                                                                       \
        int blks[processNum];                                                                            \
        std::unordered_map<int, std::string> procNameMap;                                                \
        int idx = 0;                                                                                     \
        AscendC::KernelPrintLock::GetLock();                                                             \
        AscendC::ProcessLock::GetProcessLock();                                                          \
        fflush(stdout);                                                                                  \
        for (idx = 0; idx < processNum; ++idx) {                                                         \
            set_block_dim(idx);                                                                          \
            int pid = fork();                                                                            \
            blks[idx] = pid;                                                                             \
            g_processId.push_back(pid);                                                                  \
            procNameMap.insert({ pid, AscendC::GetCoreName(idx) });                                      \
            if (pid == 0) {                                                                              \
                struct sigaction act;                                                                    \
                act.sa_handler = Handler;                                                                \
                sigemptyset(&act.sa_mask);                                                               \
                act.sa_flags = 0;                                                                        \
                sigaction(SIGILL, &act, 0);                                                              \
                sigaction(SIGBUS, &act, 0);                                                              \
                sigaction(SIGFPE, &act, 0);                                                              \
                sigaction(SIGSEGV, &act, 0);                                                             \
                sigaction(SIGPIPE, &act, 0);                                                             \
                sigaction(SIGABRT, &act, 0);                                                             \
                sigaction(SIGINT, &act, 0);                                                              \
                set_core_type(idx);                                                                      \
                AscendC::CheckGmValied(argn, kargs);                                                     \
                set_ffts_base_addr(reinterpret_cast<uint64_t>(ffts_addr));                               \
                func(__VA_ARGS__);                                                                       \
                exit(0);                                                                                 \
            }                                                                                            \
        }                                                                                                \
        struct sigaction act;                                                                            \
        act.sa_handler = Handler;                                                                        \
        sigemptyset(&act.sa_mask);                                                                       \
        act.sa_flags = 0;                                                                                \
        sigaction(SIGILL, &act, 0);                                                                      \
        sigaction(SIGBUS, &act, 0);                                                                      \
        sigaction(SIGFPE, &act, 0);                                                                      \
        sigaction(SIGSEGV, &act, 0);                                                                     \
        sigaction(SIGPIPE, &act, 0);                                                                     \
        sigaction(SIGABRT, &act, 0);                                                                     \
        sigaction(SIGINT, &act, 0);                                                                      \
        if (idx >= processNum) {                                                                         \
            for (idx = 0; idx < processNum; ++idx) {                                                     \
                int status;                                                                              \
                waitpid(blks[idx], &status, 0);                                                          \
                if (status == 0) {                                                                       \
                    std::cout << "[SUCCESS][" << procNameMap[blks[idx]] <<                               \
                        "][pid " + std::to_string(blks[idx]) + "] exit success!" << std::endl;           \
                }                                                                                        \
            }                                                                                            \
        }                                                                                                \
        AscendC::GmFree((void*)sysWorkSpacePtr);                                                         \
        g_sysWorkspaceReserved = nullptr;                                                                \
        AscendC::GmFree(ffts_addr);                                                                      \
    } while (0)

#else


inline void AscendCNpuCheckEnInterruptExit(void)
{
#ifdef ASCENDC_NPUCHK_INTER_EXIT
    AscendCEnInterruptExit();
#endif
}

#define ICPU_RUN_KF(func, blkdim, ...)                                                                   \
    do {                                                                                                 \
        g_mainPid = getpid();                                                                            \
        AscendC::CheckBlockdimForFfts(blkdim);                                                           \
        AscendC::InitSocVersion();                                                                       \
        uint64_t kargs[32];                                                                              \
        constexpr size_t workspaceSize = AscendC::RESERVED_WORKSPACE;                                    \
        uint8_t* sysWorkSpacePtr = (uint8_t*)AscendC::GmAlloc(workspaceSize);                            \
        memset_s(sysWorkSpacePtr, workspaceSize, 0, workspaceSize);                                      \
        constexpr size_t fftsCounterSize = AscendC::MAX_CORE_NUM_V220 * AscendC::MIX_IN_GROUP_CORE_NUM * \
            AscendC::FLAG_NUM * AscendC::FFTS_COUNTER_NUM;                                               \
        void* ffts_addr = AscendC::GmAlloc(fftsCounterSize);                                             \
        memset_s(ffts_addr, fftsCounterSize, 0, fftsCounterSize);                                        \
        CheckSysWorspaceSet(sysWorkSpacePtr);                                                            \
        g_sysWorkspaceReserved = sysWorkSpacePtr;                                                        \
        int32_t argn = va_args_num(__VA_ARGS__);                                                         \
        va_args_get(kargs, argn, ##__VA_ARGS__);                                                         \
        AscendC::StubInit();                                                                             \
        AscendCKernelBegin(#func, argn, kargs);                                                          \
        AscendCNpuCheckEnInterruptExit();                                                                \
        block_num = blkdim;                                                                              \
        int processNum = get_process_num();                                                              \
        g_processNum = processNum;                                                                       \
        int blks[processNum];                                                                            \
        std::unordered_map<int, std::string> procNameMap;                                                \
        int idx = 0;                                                                                     \
        AscendC::KernelPrintLock::GetLock();                                                             \
        AscendC::ProcessLock::GetProcessLock();                                                          \
        fflush(stdout);                                                                                  \
        for (idx = 0; idx < processNum; ++idx) {                                                         \
            set_block_dim(idx);                                                                          \
            int pid = fork();                                                                            \
            blks[idx] = pid;                                                                             \
            g_processId.push_back(pid);                                                                  \
            procNameMap.insert({ pid, AscendC::GetCoreName(idx) });                                      \
            if (pid == 0) {                                                                              \
                struct sigaction act;                                                                    \
                act.sa_handler = Handler;                                                                \
                sigemptyset(&act.sa_mask);                                                               \
                act.sa_flags = 0;                                                                        \
                sigaction(SIGILL, &act, 0);                                                              \
                sigaction(SIGBUS, &act, 0);                                                              \
                sigaction(SIGFPE, &act, 0);                                                              \
                sigaction(SIGSEGV, &act, 0);                                                             \
                sigaction(SIGPIPE, &act, 0);                                                             \
                sigaction(SIGABRT, &act, 0);                                                             \
                sigaction(SIGINT, &act, 0);                                                              \
                set_core_type(idx);                                                                      \
                AscendCBlockBegin((int32_t)block_idx, #func, argn, kargs);                               \
                AscendC::CheckGmValied(argn, kargs);                                                     \
                set_ffts_base_addr(reinterpret_cast<uint64_t>(ffts_addr));                               \
                try {                                                                                    \
                    func(__VA_ARGS__);                                                                   \
                } catch (std::logic_error& err) {                                                        \
                    std::cout << "[NPUCHECK ERROR]: " << err.what() <<  std::endl;                       \
                    AscendCBlockEnd((int32_t)block_idx, #func, argn, kargs);                             \
                    exit(-1);                                                                            \
                }                                                                                        \
                AscendCBlockEnd((int32_t)block_idx, #func, argn, kargs);                                 \
                exit(0);                                                                                 \
            }                                                                                            \
        }                                                                                                \
        struct sigaction act;                                                                            \
        act.sa_handler = Handler;                                                                        \
        sigemptyset(&act.sa_mask);                                                                       \
        act.sa_flags = 0;                                                                                \
        sigaction(SIGILL, &act, 0);                                                                      \
        sigaction(SIGBUS, &act, 0);                                                                      \
        sigaction(SIGFPE, &act, 0);                                                                      \
        sigaction(SIGSEGV, &act, 0);                                                                     \
        sigaction(SIGPIPE, &act, 0);                                                                     \
        sigaction(SIGABRT, &act, 0);                                                                     \
        sigaction(SIGINT, &act, 0);                                                                      \
        if (idx >= processNum) {                                                                         \
            for (idx = 0; idx < processNum; ++idx) {                                                     \
                int status;                                                                              \
                waitpid(blks[idx], &status, 0);                                                          \
                if (status == 0) {                                                                       \
                    std::cout << "[SUCCESS][" << procNameMap[blks[idx]] <<                               \
                        "][pid " + std::to_string(blks[idx]) + "] exit success!" << std::endl;           \
                }                                                                                        \
            }                                                                                            \
            AscendCKernelEnd(#func, argn, kargs);                                                        \
        }                                                                                                \
        AscendC::GmFree((void*)sysWorkSpacePtr);                                                         \
        g_sysWorkspaceReserved = nullptr;                                                                \
        AscendC::GmFree(ffts_addr);                                                                      \
        AscendC::KernelPrintLock::FreeLock();                                                            \
        AscendC::ProcessLock::FreeLock();                                                                \
    } while (0)
#endif // ASCENDC_NPUCHK_OFF
#endif // ASCENDC_ICPU_FWK_H
