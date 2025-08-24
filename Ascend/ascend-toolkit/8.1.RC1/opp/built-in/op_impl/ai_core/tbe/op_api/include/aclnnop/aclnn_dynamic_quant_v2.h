/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ACLNN_DYNAMIC_QUANT_V2_H_
#define ACLNN_DYNAMIC_QUANT_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnDynamicQuantV2GetWorkspaceSize的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDynamicQuantV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* smoothScalesOptional, const aclTensor* groupIndexOptional, int64_t dstType,
    const aclTensor* yOut, const aclTensor* scaleOut, const aclTensor* offsetOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnDynamicQuantV2的第二段接口，用于执行计算。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDynamicQuantV2(void* workspace, uint64_t workspaceSize,
                                                                       aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_DYNAMIC_QUANT_V2_H_