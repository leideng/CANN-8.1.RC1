/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http: *www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_BITWISE_XOR_TENSOR_H_
#define OP_API_INC_LEVEL2_ACLNN_BITWISE_XOR_TENSOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：计算输入张量self中每个元素与输入张量other中对应位置元素的按位异或，输入self和other必须是整数或布尔类型，对于布尔类型，
 * 计算逻辑异或。
 * 计算公式：如下
 * $$
 * \text{out}_i =
 * \text{self}_i \, \bigoplus\, \text{other}_i
 * $$
 *
 * 实现说明：如下
 * 计算图一：如下
 * 场景：经过类型推导后的数据类型为BOOL时，需要调用l0::NotEqual接口做计算
 *
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> C([l0op::NotEqual])
 *   D[(other)] --> E([l0op::Contiguous])
 *   E --> C
 *   C --> F([l0op::Cast])
 *   F --> G([l0op::ViewCopy])
 *   G --> H[(out)]
 * ```
 *
 * 计算图二：如下
 * 场景：不满足计算图一的条件时，都会调用l0::BitwiseXor接口做计算
 *
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> C([l0op::Cast])
 *   C --> D([l0op::BitwiseXor])
 *   E[(other)] --> F([l0op::Contiguous])
 *   F --> G([l0op::Cast])
 *   G --> D
 *   D --> H([l0op::Cast])
 *   H --> I([l0op::ViewCopy])
 *   I --> J[(out)]
 * ```
 */

/**
 * @brief aclnnBitwiseXorTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持BOOL、INT8、INT16、INT32、INT64、UINT8，且数据类型与other的数据类型
 * 需满足数据类型推导规则，shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in] other：npu
 * device侧的aclTensor，数据类型支持BOOL、INT8、INT16、INT32、INT64、UINT8，且数据类型与self的数据类型
 * 需满足数据类型推导规则，shape需要与self满足broadcast关系，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持BOOL、INT8、INT16、INT32、INT64、UINT8、FLOAT、FLOAT16、DOUBLE、
 * BFLOAT16、COMPLEX64、COMPLEX128，且数据类型需要是self与other推导之后可转换的数据类型，shape需要是self与other
 * broadcast之后 的shape，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBitwiseXorTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* other,
                                                            aclTensor* out, uint64_t* workspaceSize,
                                                            aclOpExecutor** executor);

/**
 * @brief aclnnBitwiseXorTensor的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnBitwiseXorTensorGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBitwiseXorTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

/**
 * @brief aclnnInplaceBitwiseXorTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] selfRef: npu device侧的aclTensor，数据类型支持BOOL、INT8、INT16、INT32、INT64、UINT8，且数据类型与other的
 * 数据类型需满足数据类型推导规则，且推导后的数据类型需要能转换成selfRef自身的数据类型，shape需要与other满足broadcast关系，且
 * broadcast之后的shape需要与selfRef自身的shape相同，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in] other：npu device侧的aclTensor，数据类型支持BOOL、INT8、INT16、INT32、INT64、UINT8，且数据类型与selfRef的
 * 数据类型需满足数据类型推导规则，shape需要与selfRef满足broadcast关系，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维
 * 以上。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseXorTensorGetWorkspaceSize(aclTensor* selfRef, const aclTensor* other,
                                                                   uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceBitwiseXorTensor的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceBitwiseXorTensorGetWorkspaceSize 获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseXorTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_BITWISE_XOR_TENSOR_H_