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
#ifndef OP_API_INC_LEVEL2_ISFINITE_H_
#define OP_API_INC_LEVEL2_ISFINITE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：判断输入张量的元素是否有界。
 * 计算图：如下
 * 场景一：非浮点数，肯定有界，直接返回True
 *
 * ```mermaid
 * graph LR
 *     A[(self)] --> C([l0op::Fill]) --> H([l0op::ViewCopy]) --> K[(out)]
 * ```
 *
 * 场景二：浮点数的情况，直接调IsFinite
 *
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Contiguous]) --> C([l0op::IsFinite]) --> H([l0op::ViewCopy]) --> K[(out)]
 * ```
 */

/**
 * @brief aclnnIsFinite的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: 原始张量。npu device侧的aclTensor，
 * 数据类型支持FLOAT、FLOAT16、BFLOAT16(仅910B支持)、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL，支持非连续的Tensor。数据格式支持ND。
 * @param [out] out: npu device侧的aclTensor，数据类型只能是BOOL，shape与self一致，支持非连续的Tensor。数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnIsFiniteGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief aclnnIsFinite的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnIsFiniteGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnIsFinite(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ISFINITE_H_
