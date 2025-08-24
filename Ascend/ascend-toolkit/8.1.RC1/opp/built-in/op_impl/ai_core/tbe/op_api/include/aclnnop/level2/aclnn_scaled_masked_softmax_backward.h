/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef OP_API_INC_LEVEL2_ACLNN_SCALED_MASKED_SOFTMAX_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_SCALED_MASKED_SOFTMAX_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnScaledMaskedSoftmaxBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：softmax的反向传播，并对结果进行缩放以及掩码。
 * 
 * @param [in] gradOutput: npu device侧的aclTensor，反向传播梯度值，数据类型支持FLOAT32，FLOAT16，BFLOAT16，
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] y: npu device侧的aclTensor，softmax函数的输出值，数据类型支持FLOAT32，FLOAT16，BFLOAT16，
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] mask: npu device侧的aclTensor，用于对计算结果进行掩码，数据类型支持Bool。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] scale: host侧的double类型，数据缩放的大小。
 * @param [in] fixedTriuMask: host侧的Bool类型，是否需要核内生成mask。
 * @param [out] out: npu device侧的aclTensor，数据类型支持FLOAT32，FLOAT16，BFLOAT16，
 * 数据类型，数据格式，tensor shape需要与self保持一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* y,
                                                                const aclTensor* mask, double scale, bool fixTriuMask,
                                                                aclTensor* out, uint64_t* workspaceSize,
                                                                aclOpExecutor** executor);

/**
 * @brief aclnnScaledMaskedSoftmaxBackward的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnScaledMaskedSoftmaxBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_SCALED_MASKED_SOFTMAX_BACKWARD_H_
