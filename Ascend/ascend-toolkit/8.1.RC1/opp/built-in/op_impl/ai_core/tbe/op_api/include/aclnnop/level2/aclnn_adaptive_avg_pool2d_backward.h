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
#ifndef OP_API_INC_AdaptiveAvgPool2dBackward_H_
#define OP_API_INC_AdaptiveAvgPool2dBackward_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAdaptiveAvgPool2dBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：aclnnAdaptiveAvgPool2d反向运算
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * @param [in] gradOutput: 正向运算结果，npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32数据类型，
 * 且数据类型与self一致。支持非连续的Tensor，数据格式支持NCHW，且数据格式需要与self一致。
 * @param [in] self: 正向输入，npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32数据类型,
 * 支持非连续的Tensor，数据格式支持NCHW
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */

ACLNN_API aclnnStatus aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                                     aclTensor* out, uint64_t* workspaceSize,
                                                                     aclOpExecutor** executor);

/**
 * @brief aclnnAdaptiveAvgPool2dBackward的第二段接口，用于执行计算。
 *
 * 算子功能：aclnnAdaptiveAvgPool2d反向运算
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAdaptiveAvgPool2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                     const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_AdaptiveAvgPool2dBackward_H_
