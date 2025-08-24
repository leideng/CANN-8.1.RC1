/**
 * @log_types.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef LOG_TYPES_H_
#define LOG_TYPES_H_

#if defined(_MSC_VER)
#define LOG_FUNC_VISIBILITY _declspec(dllexport)
#else
#define LOG_FUNC_VISIBILITY __attribute__((visibility("default")))
#endif

// log level id
#define DLOG_DEBUG 0x0      // debug level id
#define DLOG_INFO  0x1      // info level id
#define DLOG_WARN  0x2      // warning level id
#define DLOG_ERROR 0x3      // error level id

// log type
enum {
    DLOG_TYPE_DEBUG = 0,
    DLOG_TYPE_RUN = 1,
    DLOG_TYPE_MAX
};

// max log length
#define MSG_LENGTH 1024

// module id
enum {
    SLOG = 0,               /* Slog module */
    IDEDD = 1,              /* IDE daemon device */
    HCCL = 3,               /* HCCL */
    FMK = 4,                /* Adapter */
    DVPP = 6,               /* DVPP */
    RUNTIME = 7,            /* Runtime */
    CCE = 8,                /* CCE */
    HDC = 9,                /* HDC */
    DRV = 10,               /* Driver */
    DEVMM = 22,             /* Dlog memory managent */
    KERNEL = 23,            /* Kernel */
    LIBMEDIA = 24,          /* Libmedia */
    CCECPU = 25,            /* aicpu shedule */
    ROS = 27,               /* ROS */
    HCCP = 28,
    ROCE = 29,
    TEFUSION = 30,
    PROFILING =31,
    DP = 32,                /* Data Preprocess */
    APP = 33,               /* User Application */
    TS = 34,                /* Task Schedule */
    TSDUMP = 35,
    AICPU = 36,
    LP = 37,                /* Low Power */
    TDT = 38,               /* tsdaemon or aicpu shedule */
    FE = 39,
    MD = 40,
    MB = 41,
    ME = 42,
    IMU = 43,
    IMP = 44,
    GE = 45,                /* Fmk */
    CAMERA = 47,
    ASCENDCL = 48,
    TEEOS = 49,
    ISP = 50,
    SIS = 51,
    HSM = 52,
    DSS = 53,
    PROCMGR = 54,           /* Process Manager, Base Platform */
    BBOX = 55,
    AIVECTOR = 56,
    TBE = 57,
    FV = 58,
    TUNE = 60,
    HSS = 61,               /* helper */
    FFTS = 62,
    OP = 63,
    UDF = 64,
    HICAID = 65,
    TSYNC = 66,
    AUDIO = 67,
    TPRT = 68,
    ASCENDCKERNEL = 69,
    ASYS = 70,
    ATRACE = 71,
    RTC = 72,
    SYSMONITOR = 73,
    AML = 74,
    ADETECT = 75,
    INVLID_MOUDLE_ID = 76   /* add new module before INVLID_MOUDLE_ID */
};

#endif // LOG_TYPES_H_