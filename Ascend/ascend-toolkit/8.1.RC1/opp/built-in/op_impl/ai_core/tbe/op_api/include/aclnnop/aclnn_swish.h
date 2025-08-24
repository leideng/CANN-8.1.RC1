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

#ifndef OP_API_INC_LEVEL2_ACLNN_SWISH_H_
#define OP_API_INC_LEVEL2_ACLNN_SWISH_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSwish的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * 算子功能：Swish激活函数
 * @param [in] self: Device侧的aclTensor，公式中的input。支持非连续的Tensor，数据格式支持ND，self与out的shape和数据类型一致。
 * @param [in] betaOptional: Host侧的aclScalar，公式中的beta。数据类型需要是可转换为FLOAT的数据类型。
 * 当betaOptional为空指针时，默认值为1.0。
 * @param [out] out: Device侧的aclTensor，公式中的output。支持非连续的Tensor，数据格式支持ND，
 * self与out的shape和数据类型一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSwishGetWorkspaceSize(const aclTensor* self, const aclScalar* betaOptional, aclTensor* out, 
                                                uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnSwish的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAcosGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSwish(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
