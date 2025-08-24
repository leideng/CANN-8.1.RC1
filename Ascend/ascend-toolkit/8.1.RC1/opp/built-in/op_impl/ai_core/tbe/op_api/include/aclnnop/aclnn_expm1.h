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

#ifndef OP_API_INC_EXPM1_H_
#define OP_API_INC_EXPM1_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnExpm1的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：以输入的self为指数，计算自然常数e的幂，并对指数计算结果进行减1计算。对于self取值较小的场景，提供比直接用公式计算结果更高的精度。
 * 计算公式：$$ out_i = {e}^{self_i} - 1 $$
 * 实现说明-计算图
 *
 * ```mermaid
 * graph LR
 *  A[(self)] -->B([l0op::Contiguous])
 *  B --> D([l0op::Expm1])
 *  D --> H([l0op::Cast])
 *  H --> I([l0op::ViewCopy])
 *  I --> J[(out)]
 * ```
 *
 * @param [in] self：公式中输入`self`,数据类型支持INT64、BOOL、FLOAT、BFLOAT16、FLOAT16，shape需要与out一致。
 * 支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）
 * @param [out] out：公式中输入`out`,数据类型支持FLOAT、BFLOAT16、FLOAT16，shape需要与self一致。
 * 支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnExpm1GetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * @brief aclnnExpm1的第二段接口，用于执行计算
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnExpm1GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnExpm1(void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
                                 const aclrtStream stream);

/**
 * @brief aclnnInplaceExpm1的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：以输入的self为指数，计算自然常数e的幂，并对指数计算结果进行减1计算。对于self取值较小的场景，提供比直接用公式计算结果更高的精度。
 * 计算公式：$$ out_i = {e}^{self_i} - 1 $$
 * 实现说明-计算图
 *
 * ```mermaid
 * graph LR
 *  A[(selfRef)] -->B([l0op::Contiguous])
 *  B --> D([l0op::Expm1])
 *  D --> H([l0op::Cast])
 *  H --> I([l0op::ViewCopy])
 *  I --> J[(selfRef)]
 * ```
 *
 * @param [in] selfRef：公式中输入`selfRef`,数据类型支持FLOAT、FLOAT16。
 * 支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceExpm1GetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor);

/**
 * @brief aclnnInplaceExpm1的第二段接口，用于执行计算
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceExpm1GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceExpm1(void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
                                        const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EXPM1_H_
