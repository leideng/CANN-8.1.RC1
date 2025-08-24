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
#ifndef OP_API_INC_LOG10_H_
#define OP_API_INC_LOG10_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLog10的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成输入以10为底对数的计算
 * 计算公式：
 * $$ output_i=log_{10}(self_i) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0op::Contiguous])
 *     B -->C([l0op::Cast])
 *     C -->D([l0op::Log])
 *     D -->E([l0op::Cast])
 *     E -->I([l0op::ViewCopy])
 *     I -->J[(out)]
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持BOOL、INT8、INT16、INT32、INT64、UINT8、FLOAT、FLOAT16、BFLOAT16。
 *                   支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16。且数据类型是self可转化的数据类型。
 *                  数据shape与self一致。数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLog10GetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * @brief aclnnLog10的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLog10GetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLog10(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceLog10的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成输入以10为底对数的计算，在原Tensor上进行更新
 * 计算公式：
 * $$ output_i=log_{10}(self_i) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0op::Contiguous])
 *     B -->C([l0op::Cast])
 *     C -->D([l0op::Log])
 *     D -->E([l0op::Cast])
 *     E -->I([l0op::ViewCopy])
 *     I -->J[(out)]
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持BOOL、INT8、INT16、INT32、INT64、UINT8、FLOAT、FLOAT16、BFLOAT16。
 *                   支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceLog10GetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor);

/**
 * @brief aclnnInplaceLog10的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceLog10GetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceLog10(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LOG10_H_
