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
#ifndef OP_API_INC_SOFTSHRINK_H_
#define OP_API_INC_SOFTSHRINK_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSoftshrink的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 功能描述：以元素为单位，强制收缩λ范围内的元素。
 * 计算公式：如下
 * $$
 * Softshrink(x)=
 * \begin{cases}
 * x-λ, if x > λ \\
 * x+λ, if x < -λ \\
 * 0, otherwise \\
 * \end{cases}
 * $$
 * 参数描述：
 * @param [in]   self
 * 输入Tensor，数据类型支持FLOAT，FLOAT16，BFLOAT16。支持非连续Tensor，数据格式支持ND。
 * @param [in]   lambd
 * 输入Scalar，数据类型支持FLOAT。
 * @param [in]   out
 * 输出Tensor，数据类型支持FLOAT，FLOAT16，BFLOAT16。支持非连续Tensor，数据格式支持ND。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */
ACLNN_API aclnnStatus aclnnSoftshrinkGetWorkspaceSize(const aclTensor* self, const aclScalar* lambd, aclTensor* out,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnSoftshrink的第二段接口，用于执行计算。
 * 功能描述：以元素为单位，强制收缩λ范围内的元素。
 * 计算公式：如下
 * $$
 * Softshrink(x)=
 * \begin{cases}
 * x-λ, if x > λ \\
 * x+λ, if x < -λ \\
 * 0, otherwise \\
 * \end{cases}
 * $$
 * 实现说明：
 * api计算的基本路径：
```mermaid
graph LR
    A[(Self)] -->B([l0op::Contiguous])
    B -->C([l0op::SoftShrink])
    C -->D([l0op::Cast])
    D -->E([l0op::ViewCopy])
    E -->F[(Out)]

    G((lambd)) -->C
```
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSoftshrinkGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSoftshrink(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SOFTSHRINK_H_
