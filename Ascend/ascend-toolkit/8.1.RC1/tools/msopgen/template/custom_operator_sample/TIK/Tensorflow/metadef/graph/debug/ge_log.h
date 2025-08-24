/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef COMMON_GRAPH_DEBUG_GE_LOG_H_
#define COMMON_GRAPH_DEBUG_GE_LOG_H_

#include "graph/ge_error_codes.h"
#include "common/ge_common/debug/ge_log.h"
#include "common/ge_common/debug/log.h"

#define GE_LOGE(fmt, ...) GE_LOG_ERROR(GE_MODULE_NAME, ge::FAILED, fmt, ##__VA_ARGS__)

// Only check error log
#define GE_CHK_BOOL_ONLY_LOG(expr, ...) \
  do {                                  \
    const bool b = (expr);              \
    if (!b) {                           \
      GELOGI(__VA_ARGS__);              \
    }                                   \
  } while (false)

#endif  // COMMON_GRAPH_DEBUG_GE_LOG_H_

