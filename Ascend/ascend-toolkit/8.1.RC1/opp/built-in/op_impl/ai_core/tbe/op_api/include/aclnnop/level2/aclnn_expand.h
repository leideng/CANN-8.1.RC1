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
#ifndef OP_API_INC_EXPAND_H_
#define OP_API_INC_EXPAND_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnExpand的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：将输入张量self广播成指定size的张量。
 *
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Contiguous]) --> C([l0op::Expand])
 *     D((size)) --> E([sizeTensor]) --> C
 *     C --> F([l0op::ViewCopy]) --> Out[(out)]
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT、UINT8、INT8、INT32、INT64、BOOL，支持非连续的Tensor，数据格式支持ND。
 * @param [in] size: aclIntArray类型，数据类型支持INT。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT、UINT8、INT8、INT32、INT64、BOOL，需要和self保持一致，
 * 支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)），shape需要满足self的shap根据size的推导结果。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnExpandGetWorkspaceSize(const aclTensor* self, const aclIntArray* size, aclTensor* out,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnExpand的第二段接口，用于执行计算。
 *
 * 算子功能：将输入张量self广播成指定size的张量。
 *
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Contiguous]) --> C([l0op::Expand])
 *     D((size)) --> E([sizeTensor]) --> C
 *     C --> F([l0op::ViewCopy]) --> Out[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnExpandGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnExpand(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EXPAND_H_