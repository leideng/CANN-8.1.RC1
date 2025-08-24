/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef AIR_INC_FRAMEWORK_COMMON_UTIL_H_
#define AIR_INC_FRAMEWORK_COMMON_UTIL_H_
#include "common/ge_common/util.h"
namespace ge {
GE_FUNC_VISIBILITY Status ConvertToInt64(const std::string &str, int64_t &val);
GE_FUNC_VISIBILITY Status ConvertToUint64(const std::string &str, uint64_t &val);
}
#endif  // AIR_INC_FRAMEWORK_COMMON_UTIL_H_
