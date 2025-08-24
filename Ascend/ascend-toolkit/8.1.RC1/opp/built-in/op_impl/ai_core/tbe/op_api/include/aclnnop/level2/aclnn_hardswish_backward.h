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
#ifndef OP_API_INC_HARDSWISH_BACKWARD_H_
#define OP_API_INC_HARDSWISH_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnHardswishBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：激活函数hardswish的反向
 * 计算公式：
 * $$ res_{i} = grad\_output_{i} \times grad\_self_{i} $$
 * $$
 * grad\_self_{i} = \begin{cases}
 * 0, & self_{i} \lt -3, \\
 * self_{i} / 3 + 0.5, &   -3 \le self_{i} \le 3, \\
 * 1, & self_{i} \gt 3
 * \end{cases}
 * $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
graph LR
 *     A[(gradOutput)] --> B([l0op::Contiguous])
 *     B --> C([l0op::HardSwishGrad])
 *     D[(self)] --> E([l0op::Contiguous])
 *     E --> C([l0op::HardSwishGrad])
 *     C --> F([l0op::ViewCopy])
 *     F --> g[(out)]
```
 *
 * @param [in] gradOutput: npu
 * device侧的aclTensor，数据类型支持浮点类型，且数据类型需要与self一致，shape需要与self一致。
 * 支持非连续的Tensor，数据格式支持ND、NCHW、NHWC，且数据格式需要与other一致。
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持浮点类型，且数据类型需要与gradOutput一致，shape需要与gradOutput一致。
 * 支持非连续的Tensor，数据格式支持ND、NCHW、NHWC，且数据格式需要与gradOutput一致。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持浮点类型，且数据类型需要与gradOutput一致，shape需要与gradOutput一致。
 * 支持非连续的Tensor，数据格式支持ND、NCHW、NHWC，且数据格式需要与gradOutput一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnHardswishBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                             aclTensor* out, uint64_t* workspaceSize,
                                                             aclOpExecutor** executor);

/**
 * @brief aclnnHardswishBackward的第二段接口，用于执行计算。
 *
 * 算子功能：激活函数hardswish的反向
 * 计算公式：
 * $$ res_{i} = grad\_output_{i} \times grad\_self_{i} $$
 * $$
 * grad\_self_{i} = \begin{cases}
 * 0, & self_{i} \lt -3, \\
 * self_{i} / 3 + 0.5, &   -3 \le self_{i} \le 3, \\
 * 1, & self_{i} \gt 3
 * \end{cases}
 * $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
graph LR
 *     A[(gradOutput)] --> B([l0op::Contiguous])
 *     B --> C([l0op::HardSwishGrad])
 *     D[(self)] --> E([l0op::Contiguous])
 *     E --> C([l0op::HardSwishGrad])
 *     C --> F([l0op::ViewCopy])
 *     F --> g[(out)]
```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSubGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnHardswishBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_HARDSWISH_BACKWARD_H_