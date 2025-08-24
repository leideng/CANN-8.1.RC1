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
#ifndef OP_API_INC_BITWISE_NOT_H_
#define OP_API_INC_BITWISE_NOT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBitwiseNot的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：对输入的tensor进行逻辑“非”或者按位非操作

 * 计算公式：
 * $$ output_i = \lnot self_i $$
 *
 * 实现说明:
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Invert])
 *     C --> D([ViewCopy])
 *     D --> E([out])
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor, 数据类型支持INT16,INT32,INT64,INT8,UINT8,BOOL,支持非连续的Tensor,数据格式支持ND。
 * @param [in] out: npu
 * device侧的aclTensor, 数据类型支持INT16,INT32,INT64,INT8,UINT8,BOOL,且数据类型需要是self可转换的数据类型,
 * shape与self相同, 数据格式支持ND, 且数据格式需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBitwiseNotGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);
/**
 * @brief aclnnBitwiseNot的第二段接口，用于执行计算。
 *
 * 算子功能：对输入的tensor进行逻辑“非”或者按位非操作
 * 计算公式：
 * $$ output_i = \lnot self_i $$
 *
 * 实现说明:
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Invert])
 *     C --> D([ViewCopy])
 *     D --> E([out])
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnBitwiseNotGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBitwiseNot(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BITWISE_NOT_H_