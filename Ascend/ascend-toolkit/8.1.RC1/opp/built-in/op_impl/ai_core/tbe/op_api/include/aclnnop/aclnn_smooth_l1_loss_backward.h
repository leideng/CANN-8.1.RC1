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

#ifndef OP_API_INC_SMOOTH_L1_LOSS_BACKWARD_H_
#define OP_API_INC_SMOOTH_L1_LOSS_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSmoothL1LossBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能： 对输入Tensor完成SmoothL1Loss backward操作
 * 对于smoothL1Loss的第一种情况, 即$|x-y|<1$,其导数为：$$\frac{\partial SmoothL1Loss(x,y)}{\partial x} = x - y $$
 * 对于smoothL1Loss的第一种情况, 即$|x-y|\geq 1$,其导数为：$$\frac{\partial SmoothL1Loss(x,y)}{\partial x} = sign(x-y)$$
 * @param [in] gradOut: npu device侧的aclTensor, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * @param [in] self: npu device侧的aclTensor, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * @param [in] target: npu device侧的aclTensor, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * @param [in] reduction: host侧的参数，数据类型为int64_t，指定要应用到输出的缩减，支持0("none"),1("mean"),2"sum")。
 * @param [in] beta:指定在L1和L2损失之间更改的阈值，数据类型为double，该值必须是非负的。
 * @param [in] gradInput:npu device侧的aclTensor, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * 当reduction为"none"时，shape与self相同，否则shape为[1]
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码， 成功返回ACLNN_SUCCESS, 失败返回对应错误码。
 */
ACLNN_API aclnnStatus aclnnSmoothL1LossBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclTensor* self,
                                                                const aclTensor* target, int64_t reduction, float beta,
                                                                aclTensor* gradInput, uint64_t* workspaceSize,
                                                                aclOpExecutor** executor);

/**
 * @brief: aclnnSmoothL1LossBackward的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成SmoothL1Loss backward操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSigmoidBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码,成功返回ACLNN_SUCCESS, 失败返回对应错误码。
 */
ACLNN_API aclnnStatus aclnnSmoothL1LossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SMOOTH_L1_LOSS_BACKWARD_H_