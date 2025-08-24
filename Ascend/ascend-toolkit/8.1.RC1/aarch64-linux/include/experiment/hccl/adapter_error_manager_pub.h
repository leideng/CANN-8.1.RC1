/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: adapter层重构
 */

#ifndef HCCL_INC_ADAPTER_ERROR_MANAGER_PUB_H
#define HCCL_INC_ADAPTER_ERROR_MANAGER_PUB_H

#include <string>
#include <vector>

using ErrContextPub = struct Context_Pub {
    uint64_t work_stream_id;
    std::string first_stage;
    std::string second_stage;
    std::string log_header;
};

ErrContextPub hrtErrMGetErrorContextPub(void);
void hrtErrMSetErrorContextPub(ErrContextPub errorContextPub);

#ifndef HCCD
    #include "error_manager.h"
    #define RPT_INPUT_ERR(result, error_code, key, value) do { \
        if (UNLIKELY(result)) {                                           \
            REPORT_INPUT_ERROR(error_code, key, value);         \
        }                                                       \
    } while (0)

    #define RPT_ENV_ERR(result, error_code, key, value) do { \
        if (UNLIKELY(result)) {                                           \
            REPORT_ENV_ERROR(error_code, key, value);         \
        }                                                       \
    } while (0)

    constexpr char HCCL_RPT_CODE[] = "EI9999";  // 每个组件有固定的标识号码

    #define RPT_INNER_ERR_PRT(fmt, ...)                            \
        do {                                                       \
            REPORT_INNER_ERROR(HCCL_RPT_CODE, fmt, ##__VA_ARGS__); \
        } while (0)

    #define RPT_CALL_ERR(result, fmt, ...)                            \
        do {                                                          \
            if (UNLIKELY(result)) {                                             \
                REPORT_CALL_ERROR(HCCL_RPT_CODE, fmt, ##__VA_ARGS__); \
            }                                                         \
        } while (0)

    #define RPT_CALL_ERR_PRT(fmt, ...)                            \
        do {                                                      \
            REPORT_CALL_ERROR(HCCL_RPT_CODE, fmt, ##__VA_ARGS__); \
        } while (0)
#else // HCCD
    #define RPT_INPUT_ERR(result, error_code, key, value) do { \
    } while (0)

    #define RPT_ENV_ERR(result, error_code, key, value) do { \
    } while (0)

    #define RPT_INNER_ERR_PRT(fmt, ...) do { \
    } while (0)

    #define RPT_CALL_ERR(result, fmt, ...) do { \
    } while (0)

    #define RPT_CALL_ERR_PRT(fmt, ...) do { \
    } while (0)

#endif // HCCD
#endif
