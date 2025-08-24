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
#ifndef OP_API_INC_LEVEL2_ACLNN_ELU_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_ELU_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：完成ELU激活函数的反向计算，输出ELU激活函数正向输入的梯度。
 * 计算公式：无
 *
 * 实现说明：如下
 *
 * 计算图：如下
 *
 * ```mermaid
 * graph LR
 *     A[(gradOutput)] --> B([l0op::Contiguous])
 *     B --> C([l0op::EluGradV2])
 *     C --> D([l0op::Cast])
 *     D --> E([l0op::ViewCopy])
 *     E --> F[(gradInput)]
 *     G((alpha)) --> C
 *     H((scale)) --> C
 *     I((inputScale)) --> C
 *     L((isResult)) --> C
 *     J((selfOrResult)) --> K([l0op::Contiguous])
 *     K --> C
 * ```
 */

/**
 * @brief aclnnEluBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * @param [in] gradOutput：表示ELU激活函数正向输出的梯度，可自定义，类似于权重，npu
 * device侧的aclTensor，数据类型支持FLOAT、
 * FLOAT16、BFLOAT16（910B支持），支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in] alpha：表示ELU激活函数的激活系数，host侧的aclScalar，数据类型需要是可转换为FLOAT的数据类型。
 * @param [in] scale：表示ELU激活函数的缩放系数，host侧的aclScalar，数据类型需要是可转换为FLOAT的数据类型。
 * @param [in] inputScale：表示ELU激活函数的输入的缩放系数，host侧的aclScalar，数据类型需要是可转换为FLOAT的数据类型。
 * @param [in] isResult：表示传给ELU反向计算的输入是否是ELU正向的输出，数据类型支持BOOL。
 * @param [in]
 * selfOrResult：当isResult为True时，表示ELU激活函数正向的输出，当isResult为False时，表示ELU激活函数正向的输入， npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16，数据类型需要与gradOutput一致。shape需要和gradOutput的shape一致，支持
 * 非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in] gradInput：表示ELU激活函数正向输入的梯度，即对输入进行求导后的结果，npu
 * device侧的aclTensor，数据类型需要是
 * gradOutput可转换的数据类型，shape需要和gradOutput的shape一致，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEluBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclScalar* alpha,
                                                       const aclScalar* scale, const aclScalar* inputScale,
                                                       bool isResult, const aclTensor* selfOrResult,
                                                       aclTensor* gradInput, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnEluBackward的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnEluBackwardGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEluBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ELU_BACKWARD_H_
