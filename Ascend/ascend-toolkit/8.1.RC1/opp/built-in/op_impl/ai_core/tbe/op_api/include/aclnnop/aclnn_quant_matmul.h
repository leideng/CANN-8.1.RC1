
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
#ifndef OP_API_INC_QUANT_MM_H_
#define OP_API_INC_QUANT_MM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantMatmul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACL_DEPRECATED_MESSAGE("aclnnQuantMatmulGetWorkspaceSize will be deprecated, use aclnnQuantMatmulV4GetWorkspaceSize instead")
ACLNN_API aclnnStatus aclnnQuantMatmulGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                                       float deqScale, aclTensor* out, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnQuantMatmul的第二段接口，用于执行计算。
 */
ACL_DEPRECATED_MESSAGE("aclnnQuantMatmul will be deprecated, use aclnnQuantMatmulV4 instead")
ACLNN_API aclnnStatus aclnnQuantMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       const aclrtStream stream);

/**
 * @brief aclnnQuantMatmulV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACL_DEPRECATED_MESSAGE("aclnnQuantMatmulV2GetWorkspaceSize will be deprecated, use aclnnQuantMatmulV4GetWorkspaceSize instead")
ACLNN_API aclnnStatus aclnnQuantMatmulV2GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                         const aclTensor* bias, const aclTensor* deqScale, bool adjX1,
                                                         bool adjX2, aclTensor* out, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor);

/**
 * @brief aclnnQuantMatmulV2的第二段接口，用于执行计算。
 */
ACL_DEPRECATED_MESSAGE("aclnnQuantMatmulV2 will be deprecated, use aclnnQuantMatmulV4 instead")
ACLNN_API aclnnStatus aclnnQuantMatmulV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_MM_H_
