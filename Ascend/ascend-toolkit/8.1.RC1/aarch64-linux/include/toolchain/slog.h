/**
 * @slog.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2024. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef D_SYSLOG_H_
#define D_SYSLOG_H_

#include <stdarg.h>
#include <stdint.h>
#include "log_types.h"
static const int32_t TMP_LOG = 0;

#ifdef __cplusplus
#ifndef LOG_CPP
extern "C" {
#endif
#endif // __cplusplus

/**
 * @ingroup slog
 *
 * log level id
 */
#define DLOG_NULL  0x4      // don't print log
#define DLOG_EVENT 0x10     // event log print level id

/**
 * @ingroup slog
 *
 * log mask
 */
#define DEBUG_LOG_MASK      (0x00010000U)    // print log to directory debug
#define SECURITY_LOG_MASK   (0x00100000U)    // print log to directory security
#define RUN_LOG_MASK        (0x01000000U)    // print log to directory run
#define STDOUT_LOG_MASK     (0x10000000U)    // print log to stdout

#define LOG_SAVE_MODE_DEF   (0x0U)          // default
#define LOG_SAVE_MODE_UNI   (0xFE756E69U)   // unify save mode
#define LOG_SAVE_MODE_SEP   (0xFE736570U)   // separate save mode

typedef enum {
    APPLICATION = 0,
    SYSTEM
} ProcessType;

typedef struct {
    ProcessType type;       // process type
    unsigned int pid;       // pid
    unsigned int deviceId;  // device id
    unsigned int mode;      // log save mode
    char reserved[48];      // reserve 48 bytes, align to 64 bytes
} LogAttr;

/**
 * @ingroup slog
 *
 * module id
 * if a module needs to be added, add the module at the end and before INVLID_MOUDLE_ID
 */
#define ALL_MODULE  (0x0000FFFFU)

/**
 * @ingroup slog
 * @brief External log interface, which called by modules
 */
LOG_FUNC_VISIBILITY void dlog_init(void);

/**
 * @ingroup slog
 * @brief dlog_getlevel: get module debug loglevel and enableEvent
 *
 * @param [in]moduleId: moudule id(see slog.h, eg: CCE), others: invalid
 * @param [out]enableEvent: 1: enable; 0: disable
 * @return: module level(0: debug, 1: info, 2: warning, 3: error, 4: null output)
 */
LOG_FUNC_VISIBILITY int32_t dlog_getlevel(int32_t moduleId, int32_t *enableEvent);

/**
 * @ingroup slog
 * @brief dlog_setlevel: set module loglevel and enableEvent
 *
 * @param [in]moduleId: moudule id(see slog.h, eg: CCE), -1: all modules, others: invalid
 * @param [in]level: log level(0: debug, 1: info, 2: warning, 3: error, 4: null output)
 * @param [in]enableEvent: 1: enable; 0: disable, others:invalid
 * @return: 0: SUCCEED, others: FAILED
 */
LOG_FUNC_VISIBILITY int32_t dlog_setlevel(int32_t moduleId, int32_t level, int32_t enableEvent);

/**
 * @ingroup slog
 * @brief CheckLogLevel: check module level enable or not
 * users no need to call it because all dlog interface(include inner interface) has already called
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]logLevel: eg: DLOG_EVENT/DLOG_ERROR/DLOG_WARN/DLOG_INFO/DLOG_DEBUG
 * @return: 1:enable, 0:disable
 */
LOG_FUNC_VISIBILITY int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel);

/**
 * @ingroup     : slog
 * @brief       : set log attr, default pid is 0, default device id is 0, default process type is APPLICATION
 * @param [in]  : logAttrInfo   attr info, include pid(must be larger than 0), process type and device id(chip ID)
 * @return      : 0: SUCCEED, others: FAILED
 */
LOG_FUNC_VISIBILITY int32_t DlogSetAttr(LogAttr logAttrInfo);

/**
 * @ingroup     : slog
 * @brief       : print log, need va_list variable, exec CheckLogLevel() before call this function
 * @param[in]   : moduleId      module id, eg: CCE
 * @param[in]   : level         (0: debug, 1: info, 2: warning, 3: error, 16: event)
 * @param[in]   : fmt           log content
 * @param[in]   : list          variable list of log content
 */
LOG_FUNC_VISIBILITY void DlogVaList(int32_t moduleId, int32_t level, const char *fmt, va_list list);

/**
 * @ingroup slog
 * @brief DlogFlush: flush log buffer to file
 */
LOG_FUNC_VISIBILITY void DlogFlush(void);

/**
 * @ingroup slog
 * @brief dlog_error: print error log
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]fmt: log content
 */
#define dlog_error(moduleId, fmt, ...)                                          \
    do {                                                                          \
        DlogRecord(moduleId, DLOG_ERROR, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief dlog_warn: print warning log
 * call CheckLogLevel in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]fmt: log content
 */
#define dlog_warn(moduleId, fmt, ...)                                               \
    do {                                                                              \
        if (CheckLogLevel(moduleId, DLOG_WARN) == 1) {                                   \
            DlogRecord(moduleId, DLOG_WARN, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);  \
        }                                                                               \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief dlog_info: print info log
 * call CheckLogLevel in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]fmt: log content
 */
#define dlog_info(moduleId, fmt, ...)                                               \
    do {                                                                              \
        if (CheckLogLevel(moduleId, DLOG_INFO) == 1) {                                   \
            DlogRecord(moduleId, DLOG_INFO, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);  \
        }                                                                               \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief dlog_debug: print debug log
 * call CheckLogLevel in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]fmt: log content
 */
#define dlog_debug(moduleId, fmt, ...)                                              \
    do {                                                                              \
        if (CheckLogLevel(moduleId, DLOG_DEBUG) == 1) {                                  \
            DlogRecord(moduleId, DLOG_DEBUG, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                               \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief dlog_event: print event log
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]fmt: log content
 */
#define dlog_event(moduleId, fmt, ...)                                          \
    do {                                                                          \
        DlogRecord(moduleId, DLOG_EVENT, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief Dlog: print log, need caller to specify level
 * call CheckLogLevel in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]level(0: debug, 1: info, 2: warning, 3: error, 16: event)
 * @param [in]fmt: log content
 */
#define Dlog(moduleId, level, fmt, ...)                                                 \
    do {                                                                                  \
        if (CheckLogLevel(moduleId, level) == 1) {                                           \
            DlogRecord(moduleId, level, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);   \
        }                                                                                  \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief DlogSub: print log, need caller to specify level and submodule
 * call CheckLogLevel in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]submodule: eg: engine
 * @param [in]level(0: debug, 1: info, 2: warning, 3: error, 16: event)
 * @param [in]fmt: log content
 */
#define DlogSub(moduleId, submodule, level, fmt, ...)                                                   \
    do {                                                                                                  \
        if (CheckLogLevel(moduleId, level) == 1) {                                                           \
            DlogRecord(moduleId, level, "[%s:%d][%s]" fmt, __FILE__, __LINE__, submodule, ##__VA_ARGS__);    \
        }                                                                                                   \
    } while (TMP_LOG != 0)

// log interface
LOG_FUNC_VISIBILITY void DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...) __attribute((weak));

#ifdef __cplusplus
#ifndef LOG_CPP
}
#endif // LOG_CPP
#endif // __cplusplus

#ifdef LOG_CPP
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @ingroup slog
 * @brief DlogGetlevelForC: get module debug loglevel and enableEvent
 *
 * @param [in]moduleId: moudule id(see slog.h, eg: CCE), others: invalid
 * @param [out]enableEvent: 1: enable; 0: disable
 * @return: module level(0: debug, 1: info, 2: warning, 3: error, 4: null output)
 */
LOG_FUNC_VISIBILITY int DlogGetlevelForC(int moduleId, int *enableEvent);

/**
 * @ingroup slog
 * @brief DlogSetlevelForC: set module loglevel and enableEvent
 *
 * @param [in]moduleId: moudule id(see slog.h, eg: CCE), -1: all modules, others: invalid
 * @param [in]level: log level(0: debug, 1: info, 2: warning, 3: error, 4: null output)
 * @param [in]enableEvent: 1: enable; 0: disable, others:invalid
 * @return: 0: SUCCEED, others: FAILED
 */
LOG_FUNC_VISIBILITY int32_t DlogSetlevelForC(int32_t moduleId, int32_t level, int32_t enableEvent);

/**
 * @ingroup slog
 * @brief CheckLogLevelForC: check module level enable or not
 * users no need to call it because all dlog interface(include inner interface) has already called
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]logLevel: eg: DLOG_EVENT/DLOG_ERROR/DLOG_WARN/DLOG_INFO/DLOG_DEBUG
 * @return: 1:enable, 0:disable
 */
LOG_FUNC_VISIBILITY int32_t CheckLogLevelForC(int32_t moduleId, int32_t logLevel);

/**
 * @ingroup slog
 * @brief DlogSetAttrForC: set log attr, default pid is 0, default device id is 0, default process type is APPLICATION
 * @param [in]logAttrInfo: attr info, include pid(must be larger than 0), process type and device id(chip ID)
 * @return: 0: SUCCEED, others: FAILED
 */
LOG_FUNC_VISIBILITY int32_t DlogSetAttrForC(LogAttr logAttrInfo);

/**
 * @ingroup slog
 * @brief DlogForC: print log, need caller to specify level
 * call CheckLogLevelForC in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]level(0: debug, 1: info, 2: warning, 3: error, 16: event)
 * @param [in]fmt: log content
 */
#define DlogForC(moduleId, level, fmt, ...)                                                 \
    do {                                                                                  \
        if (CheckLogLevelForC(moduleId, level) == 1) {                                           \
            DlogRecordForC(moduleId, level, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);   \
        }                                                                                  \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief DlogSubForC: print log, need caller to specify level and submodule
 * call CheckLogLevelForC in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]submodule: eg: engine
 * @param [in]level(0: debug, 1: info, 2: warning, 3: error, 16: event)
 * @param [in]fmt: log content
 */
#define DlogSubForC(moduleId, submodule, level, fmt, ...)                                                   \
    do {                                                                                                  \
        if (CheckLogLevelForC(moduleId, level) == 1) {                                                           \
            DlogRecordForC(moduleId, level, "[%s:%d][%s]" fmt, __FILE__, __LINE__, submodule, ##__VA_ARGS__);    \
        }                                                                                                   \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief DlogFlushForC: flush log buffer to file
 */
LOG_FUNC_VISIBILITY void DlogFlushForC(void);

// log interface
LOG_FUNC_VISIBILITY void DlogRecordForC(int32_t moduleId, int32_t level, const char *fmt, ...) __attribute((weak));

#ifdef __cplusplus
}
#endif
#endif // LOG_CPP
#endif // D_SYSLOG_H_
