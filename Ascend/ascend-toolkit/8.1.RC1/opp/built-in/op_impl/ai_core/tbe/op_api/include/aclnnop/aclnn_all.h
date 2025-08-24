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
#ifndef OP_API_INC_LEVEL2_ACLNN_ALL_H_
#define OP_API_INC_LEVEL2_ACLNN_ALL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACLNN_MAX_SHAPE_RANK 8
#define DIM_BITS_LEN 64

/**
 * @brief aclnnAll的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：对Tensor中的所有元素进行与运算。

 *
 * 计算图：
 * ```mermaid
 * graph LR
 *   A[(self)] -->B([l0::Contiguous])
 *   B --> E([l0::ReduceAll])
 *   C[(dim)] --> E
 *   D[(keepdim)] --> E
 *   E --> G([l0::ViewCopy])
 *   G --> H[(out)]
 * ```
 *
 * @param [in] self: 待进行all计算的入参。npu device侧的aclTensor，
 * 数据类型支持BOOL，数据格式支持ND，支持非连续的Tensor。
 * @param [in] dim: 需要压缩的维度，值需要在输入Tensor范围内，支持负数，数据类型支持INT
 * @param [in] keepdim: 输出张量`dim`是否与输入保持一致，数据类型支持BOOL
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAllGetWorkspaceSize(const aclTensor* self, const aclIntArray* dim, bool keepdim,
                                               aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAll的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAllGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAll(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                               const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ALL_H_