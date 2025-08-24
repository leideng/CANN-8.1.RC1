/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_FRAMEWORK_COMMON_GE_PROFILING_H_
#define INC_FRAMEWORK_COMMON_GE_PROFILING_H_

#include "external/ge/ge_api_error_codes.h"
#include "runtime/base.h"

/// @brief Output the profiling data of single operator in Pytorch, and does not support multithreading
/// @return Status result
GE_FUNC_VISIBILITY ge::Status ProfSetStepInfo(const uint64_t index_id, const uint16_t tag_id, rtStream_t const stream);

GE_FUNC_VISIBILITY ge::Status ProfGetDeviceFormGraphId(const uint32_t graph_id, uint32_t &device_id);

#endif  // INC_FRAMEWORK_COMMON_GE_PROFILING_H_
