/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: wrapper header
 * Author: Huawei Technologies Co., Ltd.
 * Create: 2024-12-16
 */
#ifndef MSSERVICEPROFILER_H
#define MSSERVICEPROFILER_H

#include "Profiler.h"

#define SERVER_PROFILER
#define PRIVATE_MACRO_VAR_ARGS_IMPL_COUNT(_1, _2, N, ...) N
#define PRIVATE_MACRO_VAR_ARGS_IMPL(args) PRIVATE_MACRO_VAR_ARGS_IMPL_COUNT args
#define PRIVATE_COUNT_MACRO_VAR_ARGS(...) PRIVATE_MACRO_VAR_ARGS_IMPL((__VA_ARGS__, 2, 1))

#define PRIVATE_MACRO_CHOOSE_HELPER1(M, count) M##count
#define PRIVATE_MACRO_CHOOSE_HELPER(M, count) PRIVATE_MACRO_CHOOSE_HELPER1(M, count)

#define PRIVATE_PROF_STMT(_STMT) _STMT
#define PRIVATE_PROF_STMT_LEVEL(_LEVEL, _STMT) msServiceProfiler::Profiler<msServiceProfiler::_LEVEL>()._STMT

#define PRIVATE_PROF1 PRIVATE_PROF_STMT
#define PRIVATE_PROF2 PRIVATE_PROF_STMT_LEVEL

#define PROF(...) PRIVATE_MACRO_CHOOSE_HELPER(PRIVATE_PROF, PRIVATE_COUNT_MACRO_VAR_ARGS(__VA_ARGS__))(__VA_ARGS__)
#define MONITOR(...) PROF(__VA_ARGS__)

#endif  // MSSERVICEPROFILER_H
