/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_BASE_ERR_MSG_H_
#define INC_EXTERNAL_BASE_ERR_MSG_H_

#include <cstdint>
#include <vector>

#include "graph/types.h"

#define REPORT_INNER_ERR_MSG(error_code, format, ...) \
  (void)ge::ReportInnerErrMsg(__FILE__, __FUNCTION__, __LINE__, (error_code), (format), ##__VA_ARGS__)

namespace ge {

/**
 * Report inner error message
 * @param [in] file_name: report file name
 * @param [in] func: report function name
 * @param [in] line: report line number of file_name
 * @param [in] error_code: predefined error code
 * @param [in] format: format of error message
 * @param [in] ...: value of arguments
 */

GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY
int32_t ReportInnerErrMsg(const char *file_name, const char *func, uint32_t line, const char *error_code,
                          const char *format, ...) FORMAT_PRINTF(5, 6) WEAK_SYMBOL;

/**
 * Report user defined error message
 * @param [in] error_code: user defined error code, support EU0000 ~ EU9999
 * @param [in] format: format of error message
 * @param [in] ...: value of arguments
 * @return int32_t 0(success) -1(fail)
 */
GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY
int32_t ReportUserDefinedErrMsg(const char *error_code, const char *format, ...) FORMAT_PRINTF(2, 3) WEAK_SYMBOL;

/**
 * Report CANN predefined error message
 * @param [in] error_code: predefined error code
 * @param [in] key: vector parameter key
 * @param [in] value: vector parameter value
 * @return int32_t 0(success) -1(fail)
 */
GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY
int32_t ReportPredefinedErrMsg(const char *error_code, const std::vector<const char *> &key,
                               const std::vector<const char *> &value) WEAK_SYMBOL;

}  // namespace ge

#endif  // INC_EXTERNAL_BASE_ERR_MSG_H_
