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
#ifndef OP_API_INC_LEVEL2_ACLNN_EQ_SCALAR_H_
#define OP_API_INC_LEVEL2_ACLNN_EQ_SCALAR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnEqScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 计算self中的元素的值与other的值是否相等，将self每个元素与other的值的比较结果写入out中：
 *
 * $$ out_i = (self_i == \mathit{other} )  ?  [True] : [False] $$
 *
 *
 * 计算图：
 * ```mermaid
 * graph LR
 * A[(Self)] -->B([l0op::Contiguous])
 * B -->C1([l0op::Cast])-->D([l0op::Equal])
 *     F((other)) -->D
 *     D -->F1([l0op::ViewCopy])--> J[(out)]
 * ```
 * @param [in] self: npu device侧的aclTensor，
 * 数据类型支持FLOAT16, FLOAT, INT64, UINT64, INT32, INT8, UINT8, BOOL, UINT32, BFLOAT16, DOUBLE, INT16, COMPLEX64,
 * COMPLEX128，数据格式支持ND，支持非连续的Tensor。
 * @param [in] other: host侧的aclScalar，数据类型支持FLOAT16, FLOAT, INT64, UINT64, INT32, INT8, UINT8, BOOL, UINT32,
 * BFLOAT16, DOUBLE, INT16, COMPLEX64, COMPLEX128
 * @param [in] out: npu device侧的aclTensor，数据类型为bool，shape与self的shape一致，数据格式支持ND
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEqScalarGetWorkspaceSize(const aclTensor* self, const aclScalar* other, aclTensor* out,
                                                    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnEqScalar的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnEqScalarGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEqScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    const aclrtStream stream);

/**
 * @brief aclnnInplaceEqScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 计算self中的元素的值与other的值是否相等，将self每个元素与other的值的比较结果原地返回到self中：
 *
 * $$ self_i = (self_i == \mathit{other} )  ?  [True] : [False] $$
 *
 * 计算图：
*```mermaid
* graph LR
* A[(SelfRef)] -->B([l0op::Contiguous])
* B -->C1([l0op::Cast])-->D([l0op::Equal])
* F((other)) -->D
* D -->F1([l0op::ViewCopy])--> J[(selfRef)]
```
 * @param [in] selfRef: npu device侧的aclTensor，
 * 数据类型支持FLOAT16, FLOAT, INT64, UINT64, INT32, INT8, UINT8, BOOL, UINT32, BFLOAT16, DOUBLE, INT16, COMPLEX64,
 * COMPLEX128，数据格式支持ND，支持非连续的Tensor。
 * @param [in] other: host侧的aclScalar，数据类型支持FLOAT16, FLOAT, INT64, UINT64, INT32, INT8, UINT8, BOOL, UINT32,
 * BFLOAT16, DOUBLE, INT16, COMPLEX64, COMPLEX128
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceEqScalarGetWorkspaceSize(const aclTensor* selfRef, const aclScalar* other,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceEqScalar的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnEqScalarGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceEqScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_EQ_SCALAR_H_