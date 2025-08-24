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
#ifndef OP_API_INC_BITWISE_AND_TENSOR_H_
#define OP_API_INC_BITWISE_AND_TENSOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBitwiseAndTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成位与或者逻辑与计算
 * 计算公式：
 * $$ output_i = self_i\&other_i $$
 *
 * 实现说明
 * 计算图一
 * 场景一：经过类型推导后，self和other的数据类型都为BOOL类型时，需要调用l0::LogicalAnd接口做计算：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([LogicalAnd])
 *     D[(other)] -->E([Contiguous])
 *     E --> F([Cast])
 *     F --> G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 * 计算图二
 * 场景二：经过类型推导后，self和other的数据类型都为INT类型时，需要调用l0::BitwiseAnd接口做计算：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([BitwiseAnd])
 *     D[(other)] -->E([Contiguous])
 *     E --> F([Cast])
 *     F --> G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持INT16,UINT16,INT32,INT64,INT8,UINT8,BOOL，且数据类型需要与other构成互相推导关系，
 * shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: npu
 * device侧的aclTensor，数据类型支持INT16,UINT16,INT32,INT64,INT8,UINT8,BOOL，且数据类型需要与self构成互相推导关系，
 * shape需要与self满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持INT16,UINT16,INT32,INT64,INT8,UINT8,BOOL，且数据类型需要是self与other推导之后可转换的数据类型，
 * shape需要是self与other，broadcast之后的shape，数据格式支持ND，且数据格式需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBitwiseAndTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* other,
                                                            aclTensor* out, uint64_t* workspaceSize,
                                                            aclOpExecutor** executor);

/**
 * @brief aclnnBitwiseAndTensor的第二段接口，用于执行计算。
 *
 * 算子功能：完成位与或者逻辑与计算
 * 计算公式：
 * $$ output_i = self_i\&other_i $$
 *
 * 实现说明：
 * 计算图一
 * 场景一：经过类型推导后，self和other的数据类型都为BOOL类型时，需要调用l0::LogicalAnd接口做计算：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([LogicalAnd])
 *     D[(other)] -->E([Contiguous])
 *     E --> F([Cast])
 *     F --> G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 * 计算图二
 * 场景二：经过类型推导后，self和other的数据类型都为INT类型时，需要调用l0::BitwiseAnd接口做计算：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([BitwiseAnd])
 *     D[(other)] -->E([Contiguous])
 *     E --> F([Cast])
 *     F --> G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnBitwiseAndTensorGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBitwiseAndTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

/**
 * @brief aclnnInplaceBitwiseAndTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成位与或者逻辑与计算
 * 计算公式：
 * $$ output_i = self_i\&other_i $$
 *
 * 实现说明：
 * 计算图一
 * 场景一：经过类型推导后，self和other的数据类型都为BOOL类型时，需要调用l0::LogicalAnd接口做计算：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([LogicalAnd])
 *     D[(other)] -->E([Contiguous])
 *     E --> F([Cast])
 *     F --> G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 * 计算图二
 * 场景二：经过类型推导后，self和other的数据类型都为INT类型时，需要调用l0::BitwiseAnd接口做计算：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([BitwiseAnd])
 *     D[(other)] -->E([Contiguous])
 *     E --> F([Cast])
 *     F --> G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持INT16,UINT16,INT32,INT64,INT8,UINT8,BOOL，且数据类型需要与other构成互相推导关系，
 * shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: npu
 * device侧的aclTensor，数据类型支持INT16,UINT16,INT32,INT64,INT8,UINT8,BOOL，且数据类型需要与self构成互相推导关系，
 * shape需要与self满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseAndTensorGetWorkspaceSize(const aclTensor* selfRef, const aclTensor* other,
                                                                   uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceBitwiseAndTenosr的第二段接口，用于执行计算。
 *
 * 算子功能：完成位与或者逻辑与计算
 * 计算公式：
 * $$ output_i = self_i\&other_i $$
 *
 * 实现说明：
 * 计算图一
 * 场景一：经过类型推导后，self和other的数据类型都为BOOL类型时，需要调用l0::LogicalAnd接口做计算：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([LogicalAnd])
 *     D[(other)] -->E([Contiguous])
 *     E --> F([Cast])
 *     F --> G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 * 计算图二
 * 场景二：经过类型推导后，self和other的数据类型都为INT类型时，需要调用l0::BitwiseAnd接口做计算：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([BitwiseAnd])
 *     D[(other)] -->E([Contiguous])
 *     E --> F([Cast])
 *     F --> G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceBitwiseAndTensorOutGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseAndTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BITWISE_AND_TENSOR_H_
