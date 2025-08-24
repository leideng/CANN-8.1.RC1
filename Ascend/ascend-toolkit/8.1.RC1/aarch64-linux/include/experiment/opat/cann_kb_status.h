/**
 * @file file cann_kb_status.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef CANN_KB_STATUS_H
#define CANN_KB_STATUS_H

namespace CannKb {
enum class CANN_KB_STATUS : int {              /* < CANN_KB Status Define > */
    CANN_KB_SUCC = 0,                          // CANN_KB SUCCESS
    CANN_KB_FAILED,                            // CANN_KB FAILED
    CANN_KB_CHECK_FAILED,                      // CANN_KB check failed
    CANN_KB_INIT_ERR,                          // CANN_KB init ERROR
    CANN_KB_GET_PARAM_ERR,                        // CANN_KB get param ERROR
    CANN_KB_PY_NULL,                     // CANN_KB python return null
    CANN_KB_PY_FAILED,                         // CANN_KB python return failed
};
} // namespace CannKb
#endif
