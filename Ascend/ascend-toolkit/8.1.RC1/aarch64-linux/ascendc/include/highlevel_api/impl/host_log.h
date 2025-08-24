/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file host_log.h
 * \brief
 */

#ifndef IMPL_HOST_LOG_H
#define IMPL_HOST_LOG_H
#include <cassert>
#include <cstdint>
#include "toolchain/slog.h"
#include "alog_pub.h"

#define ASCENDC_MODULE_NAME static_cast<int32_t>(ASCENDCKERNEL)

#define ASCENDC_HOST_ASSERT(cond, ret, format, ...)                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      if (AlogRecord == nullptr) {                                             \
        dlog_error(ASCENDC_MODULE_NAME, "[%s] " format "\n", __FUNCTION__,     \
                   ##__VA_ARGS__);                                             \
      } else {                                                                 \
        if (AlogCheckDebugLevel(ASCENDC_MODULE_NAME, DLOG_ERROR) == 1) {       \
          AlogRecord(ASCENDC_MODULE_NAME, DLOG_TYPE_DEBUG, DLOG_ERROR,         \
                     "[%s:%d][%s] " format "\n", __FILE__, __LINE__,           \
                     __FUNCTION__, ##__VA_ARGS__);                             \
        }                                                                      \
      }                                                                        \
      ret;                                                                     \
    }                                                                          \
  } while (0)

// 0 debug, 1 info, 2 warning, 3 error
#define TILING_LOG_ERROR(format, ...)                                          \
  do {                                                                         \
    if (AlogRecord == nullptr) {                                               \
      dlog_error(ASCENDC_MODULE_NAME, "[%s] " format "\n", __FUNCTION__,       \
                 ##__VA_ARGS__);                                               \
    } else {                                                                   \
      if (AlogCheckDebugLevel(ASCENDC_MODULE_NAME, DLOG_ERROR) == 1) {         \
        AlogRecord(ASCENDC_MODULE_NAME, DLOG_TYPE_DEBUG, DLOG_ERROR,           \
                   "[%s:%d][%s] " format "\n", __FILE__, __LINE__,             \
                   __FUNCTION__, ##__VA_ARGS__);                               \
      }                                                                        \
    }                                                                          \
  } while (0)
#define TILING_LOG_INFO(format, ...)                                           \
  do {                                                                         \
    if (AlogRecord == nullptr) {                                               \
      dlog_info(ASCENDC_MODULE_NAME, "[%s] " format "\n", __FUNCTION__,        \
                ##__VA_ARGS__);                                                \
    } else {                                                                   \
      if (AlogCheckDebugLevel(ASCENDC_MODULE_NAME, DLOG_INFO) == 1) {          \
        AlogRecord(ASCENDC_MODULE_NAME, DLOG_TYPE_DEBUG, DLOG_INFO,            \
                   "[%s:%d][%s] " format "\n", __FILE__, __LINE__,             \
                   __FUNCTION__, ##__VA_ARGS__);                               \
      }                                                                        \
    }                                                                          \
  } while (0)
#define TILING_LOG_WARNING(format, ...)                                        \
  do {                                                                         \
    if (AlogRecord == nullptr) {                                               \
      dlog_warn(ASCENDC_MODULE_NAME, "[%s] " format "\n", __FUNCTION__,        \
                ##__VA_ARGS__);                                                \
    } else {                                                                   \
      if (AlogCheckDebugLevel(ASCENDC_MODULE_NAME, DLOG_WARN) == 1) {          \
        AlogRecord(ASCENDC_MODULE_NAME, DLOG_TYPE_DEBUG, DLOG_WARN,            \
                   "[%s:%d][%s] " format "\n", __FILE__, __LINE__,             \
                   __FUNCTION__, ##__VA_ARGS__);                               \
      }                                                                        \
    }                                                                          \
  } while (0)
#define TILING_LOG_DEBUG(format, ...)                                          \
  do {                                                                         \
    if (AlogRecord == nullptr) {                                               \
      dlog_debug(ASCENDC_MODULE_NAME, "[%s] " format "\n", __FUNCTION__,       \
                 ##__VA_ARGS__);                                               \
    } else {                                                                   \
      if (AlogCheckDebugLevel(ASCENDC_MODULE_NAME, DLOG_DEBUG) == 1) {         \
        AlogRecord(ASCENDC_MODULE_NAME, DLOG_TYPE_DEBUG, DLOG_DEBUG,           \
                   "[%s:%d][%s] " format "\n", __FILE__, __LINE__,             \
                   __FUNCTION__, ##__VA_ARGS__);                               \
      }                                                                        \
    }                                                                          \
  } while (0)
#endif // IMPL_HOST_LOG_H
