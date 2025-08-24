/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_CXX_ASCEND_LIMITS_H
#define METADEF_CXX_ASCEND_LIMITS_H
#include <cstdint>
#include <cstddef>

namespace ge {
constexpr uint32_t kDefaultMaxInputNum = 8U;
constexpr uint32_t kDefaultMaxOutputNum = 8U;
constexpr size_t kDefaultDimsNum = 8U;
constexpr size_t kMaxNameLen = 1024UL;
}
#endif  // METADEF_CXX_ASCEND_LIMITS_H
