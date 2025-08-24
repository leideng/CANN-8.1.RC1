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

#ifndef OP_API_INC_HARDSHRINK_BACKWARD_H_
#define OP_API_INC_HARDSHRINK_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnHardshrinkBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能： 对输入Tensor完成hardShrink backward操作
 * @param [in] gradOutput: 计算输入, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * @param [in] self: 计算输入, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * @param [in] lambd:计算输入, 指定hardShrinkBackward分段的阈值，数据类型支持FLOAT类型。
 * @param [in] gradInput:计算输出, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码， 成功返回ACLNN_SUCCESS, 失败返回对应错误码。
 */
ACLNN_API aclnnStatus aclnnHardshrinkBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                              const aclScalar* lambd, aclTensor* gradInput,
                                                              uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnHardshrinkBackward的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成hardShrink backward操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnHardshrinkBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码,成功返回ACLNN_SUCCESS, 失败返回对应错误码。
 */
ACLNN_API aclnnStatus aclnnHardshrinkBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_HARDSHRINK_BACKWARD_H_