/**
 * @file awatchdog_types.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef AWATCHDOG_TYPES_H
#define AWATCHDOG_TYPES_H
#include <stdint.h>

#define AWD_EXPORT __attribute__((visibility("default")))
typedef void (*AwatchdogCallbackFunc) (void *);
typedef int32_t  AwdStatus;
typedef intptr_t AwdHandle;

// value of AwdStatus
#define AWD_SUCCESS              0
#define AWD_FAILURE              (-1)
#define AWD_INVLIAD_PARAM        (-2)

// value of AwdHandle
#define AWD_INVALID_HANDLE      (-1)
#define AWD_ALL_HANDLE          (0)

// watchdog status
#define AWD_STATUS_DESTROYED -2
#define AWD_STATUS_INIT -1
#define AWD_STATUS_STARTED 0

typedef struct AwdThreadWatchdog {
    uint32_t dogId;          // [31:16] watchdog type, [15:0] moduleId
    uint32_t timeout;        // timeout, s
    int32_t runCount;
    int32_t startCount;
    int32_t tid;
    int32_t pid;
    AwatchdogCallbackFunc callback;
} AwdThreadWatchdog;

#endif