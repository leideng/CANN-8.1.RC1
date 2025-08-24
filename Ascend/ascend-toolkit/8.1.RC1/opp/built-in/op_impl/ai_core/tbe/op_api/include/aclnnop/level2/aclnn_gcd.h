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

#ifndef OP_API_INC_GCD_H_
#define OP_API_INC_GCD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGcd的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：对给定的self和other计算element-wise维度的最大公约数。
 *
 * @param [in] self:
 * 输入`self`，与other满足数据类型推导规则，推导后数据类型支持INT32、INT16(Ascend910B)，shape需要与other满足broadcast规则。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] other:
 * 输入`other`，与self满足数据类型推导规则，推导后数据类型支持INT32、INT16(Ascend910B)，shape需要与self满足broadcast规则。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: 输出`out`，数据类型支持INT32、INT16(Ascend910B)。shape需要与self和other进行broadcast后的shape一致，
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGcdGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                               uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGcd的第二段接口，用于执行计算。
 *
 * 算子功能：对给定的self和other计算element-wise维度的最大公约数。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGcdGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGcd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GCD_H_
