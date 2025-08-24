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
#ifndef OP_API_INC_RSUB_H_
#define OP_API_INC_RSUB_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRsubs的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成减法计算
 * 计算公式：
 * $$ out_i = other - self_i * alpha $$
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持整型，浮点类型，复数，且数据类型需要与other构成互相推导关系，
 * shape需要与other满足broadcast关系。支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: host侧的aclScalar，数据类型支持整型，浮点类型，复数，且数据类型需要与self构成互相推导关系。
 * @param [in] alpha: host侧的aclScalar，数据类型需要可转换成self与other推导后的数据类型。
 * @param [in] out: npu device侧的aclTensor，数据类型支持整型，浮点类型，复数，
 * 且数据类型需要是self与other推导之后可转换的数据类型，shape与self一致。支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRsubsGetWorkspaceSize(const aclTensor* self, const aclScalar* other, const aclScalar* alpha,
                                                 aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnRsubs的第二段接口，用于执行计算。
 *
 * 算子功能：完成减法计算
 * 计算公式：
 * $$ out_i = other - self_i * alpha $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnRsubsGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRsubs(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnRsub的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成减法计算
 * 计算公式：
 * $$ out_i = other_i - self_i * alpha $$
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持整型，浮点类型，复数，且数据类型需要与other构成互相推导关系，
 * shape需要与other满足broadcast关系。支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: npu device侧的aclTensor，数据类型支持整型，浮点类型，复数，且数据类型需要与self构成互相推导关系，
 * shape需要与self满足broadcast关系。支持非连续的Tensor，数据格式支持ND。
 * @param [in] alpha: host侧的aclScalar，数据类型需要可转换成self与other推导后的数据类型。
 * @param [in] out: npu device侧的aclTensor，数据类型支持整型，浮点类型，复数，
 * 且数据类型需要是self与other推导之后可转换的数据类型，shape需要是self与other broadcast之后的shape。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRsubGetWorkspaceSize(const aclTensor* self, const aclTensor* other, const aclScalar* alpha,
                                                aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnRsub的第二段接口，用于执行计算。
 *
 * 算子功能：完成加法计算
 * 计算公式：
 * $$ out_i = other_i - self_i * alpha $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnRsubGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRsub(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_RSUB_H_
