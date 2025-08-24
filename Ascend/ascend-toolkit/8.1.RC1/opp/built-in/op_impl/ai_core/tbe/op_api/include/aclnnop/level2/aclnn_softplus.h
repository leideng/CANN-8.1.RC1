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
#ifndef OP_API_INC_SOFTPLUS_H_
#define OP_API_INC_SOFTPLUS_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSoftplus的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：激活函数softplus
 * 计算公式：
 * $$
 * Softplus(x) = \begin{cases}
 * \frac{1}{\beta} \log(1+\exp(\beta x)), \beta *x \le threshold \\
 * x, \beta *x > threshold
 * \end{cases}
 * $$
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持浮点类型，支持非连续的Tensor，数据格式支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW。
 * @param [in] beta: host侧的aclScalar，数据类型需要可转换成self的数据类型。
 * @param [in] threshold：host侧的aclScalar，数据类型需要可转换成self的数据类型。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持浮点类型，支持非连续的Tensor，数据格式支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSoftplusGetWorkspaceSize(const aclTensor* self, const aclScalar* beta,
                                                    const aclScalar* threshold, aclTensor* out, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief aclnnSoftplus的第二段接口，用于执行计算。
 *
 * 算子功能：激活函数softplus
 * 计算公式：
 * $$
 * Softplus(x) = \begin{cases}
 * \frac{1}{\beta} \log(1+\exp(\beta x)), \beta *x \le threshold \\
 * x, \beta *x > threshold
 * \end{cases}
 * $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSoftplusGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSoftplus(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SOFTPLUS_H_