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
#ifndef OP_API_INC_TANH_BACKWARD_H_
#define OP_API_INC_TANH_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnTanhBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：激活函数tanh(x)的反向
 *
 * 实现说明：api
 * 计算的基本路径：如下所示
 * ```mermaid
 * graph LR
 *         A[(gradOutput)] -->B([l0op::Contiguous]) --> C([l0op::TanhGrad])
 *         D[(output)] -->E([l0op::Contiguous]) --> C
 *         C --> F([l0op::ViewCopy])
 *         F --> G[(gradInput)]
 * ```
 *
 * @param [in] gradOutput: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、DOUBLE，
 * 且数据类型与output一致,shape与output相同。支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND。
 * @param [in] output: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、DOUBLE。
 * 支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND。
 * @param [out] gradInput: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、DOUBLE。
 * 支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnTanhBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* output,
                                                        aclTensor* gradInput, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor);

/**
 * @brief aclnnTanhBackward的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnTanhBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnTanhBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TANH_BACKWARD_H_
