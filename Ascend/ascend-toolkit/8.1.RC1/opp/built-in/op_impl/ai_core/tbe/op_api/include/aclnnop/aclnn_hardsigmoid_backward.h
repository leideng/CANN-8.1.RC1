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
#ifndef OP_API_INC_HARDSIGMOID_BACKWARD_H_
#define OP_API_INC_HARDSIGMOID_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnHardsigmoidBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：激活函数hardsigmoid的反向计算。
 *
 * @param [in] gradOutput(aclTensor*, 计算输入): 数据类型支持FLOAT、FLOAT16，支持非连续的Tensor，数据格式支持ND。
 * @param [in] self(aclTensor*, 计算输入): 数据类型支持FLOAT、FLOAT16，支持非连续的Tensor，数据格式支持ND。
 * @param [out] out(aclTensor*, 计算输出): 数据类型支持FLOAT、FLOAT16，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspace_size(uint64_t*, 出参): 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor(aclOpExecutor**, 出参): 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnHardsigmoidBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                               aclTensor* out, uint64_t* workspaceSize,
                                                               aclOpExecutor** executor);

/**
 * @brief aclnnHardsigmoidBackward的第二段接口，执行计算。
 *
 * 算子功能：激活函数hardsigmoid的反向计算。
 *
 * @param [in] workspace(void*, 入参): 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize(uint64_t, 入参): 在npu
 * device侧申请的workspace大小，由第一段接口aclnnHardsigmoidBackwardGetWorkspaceSize获取。
 * @param [in] executor(aclOpExecutor*, 入参): op执行器，包含了算子计算流程。
 * @param [in] stream(aclrtStream, 入参): acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnHardsigmoidBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_HARDSIGMOID_BACKWARD_H_
