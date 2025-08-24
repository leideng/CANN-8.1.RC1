/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/license/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_REAL_H_
#define OP_API_INC_LEVEL2_ACLNN_REAL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACLNN_MAX_SHAPE_RANK 8

/**
 * @brief aclnnReal的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：返回输入Tensor中每个元素绝对值的结果
 * 计算公式：
 * $$ out_{i} = real(self_{i}) $$
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *   A[(self)]--->B([l0op::Contiguous])
 *   B--->C([l0op::Real])
 *   C--->D([l0op::ViewCopy])
 *   D--->E[(out)]
 * ```
 *
 * @param [in] self: 待进行real计算的入参。npu device侧的aclTensor,
 * 数据类型支持FLOAT、FLOAT16、COMPLEX32、COMPLEX64、COMPLEX128，数据格式支持ND，支持非连续的Tensor。
 * @param [in] out: real计算的出参。npu device侧的aclTensor,
 * 数据类型支持FLOAT、FLOAT16、DOUBLE，数据格式支持ND，支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包括算子计算流程
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRealGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief aclnnReal的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnRealGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnReal(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_REAL_H_
