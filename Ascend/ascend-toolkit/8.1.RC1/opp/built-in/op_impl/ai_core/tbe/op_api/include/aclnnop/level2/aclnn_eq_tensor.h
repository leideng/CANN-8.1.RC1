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
#ifndef OP_API_INC_EQTENSOR_H_
#define OP_API_INC_EQTENSOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnEqTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 计算两个Tensor中的元素是否相等，返回一个Tensor，self=other的为True(1.)，否则为False(0.)：
 *
 * $$ out = (self == other)  ?  [True] : [False] $$
 *
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *    A[(Self)] -->B([l0op::Contiguous])
 *    B -->C1([l0op::Cast])-->D([l0op::Equal])
 *    E[(other)] -->F([l0op::Contiguous])
 *    F --> C2([l0op::Cast])-->D
 *    D --> C3([l0op::Cast])-->F1([l0op::ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT16,FLOAT,INT64,INT32,INT8,UINT8,BOOL,BFLOAT16,
 * DOUBLE,INT16,COMPLEX64,COMPLEX128数据类型，shape需要与other满足broadcast关系，
 * 数据类型需要与other满足数据类型推导规则，支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: npu device侧的aclTensor，数据类型支持FLOAT16,FLOAT,INT64,INT32,INT8,UINT8,BOOL,BFLOAT16,
 * DOUBLE,INT16,COMPLEX64,COMPLEX128数据类型，shape需要与self满足broadcast关系，
 * 数据类型需要与self满足数据类型推导规则，支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT16,FLOAT,INT64,INT32,INT8,UINT8,BOOL,BFLOAT16,
 * DOUBLE,INT16,COMPLEX64,COMPLEX128，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEqTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                                    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnEqTensor的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnEqTensorGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEqTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

/**
 * @brief aclnnInplaceEqTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：计算selfRef中元素的值与other的值是否相等，将selfRef每个元素与other的值的比较结果原地返回到selfRef中；
 *
 * 计算公式：$$ selfRef = (selfRef == other)  ?  [True] : [False] $$
 *
 * @param [in] selfRef: npu device侧的aclTensor，
 * 数据类型支持FLOAT16,FLOAT,INT64,INT32,INT8,UINT8,BOOL,BFLOAT16(仅1971支持),DOUBLE,INT16,COMPLEX64,COMPLEX128，
 * 数据格式支持ND，支持非连续的Tensor。
 * @param [in] other: npu device侧的aclTensor，
 * 数据类型支持FLOAT16,FLOAT,INT64,INT32,INT8,UINT8,BOOL,BFLOAT16(仅1971支持) ,DOUBLE,INT16,COMPLEX64,COMPLEX128，
 * 数据格式支持ND，支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceEqTensorGetWorkspaceSize(const aclTensor* selfRef, const aclTensor* other,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceEqTensor的第二段接口，用于执行计算。
 *
 * 算子功能：计算selfRef中元素的值与other的值是否相等，将selfRef每个元素与other的值的比较结果原地返回到selfRef中；
 *
 * 计算公式：$$ selfRef = (selfRef == other)  ?  [True] : [False] $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceAddGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceEqTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EQTENSOR_H_