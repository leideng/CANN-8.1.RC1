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
#ifndef OP_API_INC_MASKEF_FILL_SCALAR_H_
#define OP_API_INC_MASKEF_FILL_SCALAR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInplaceMaskedFillScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：用value填充selfRef里面与mask矩阵中值为true的位置相对应的元素
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *  A[(selfRef)] -->B([Contiguous])
 *  B  -->C([Unsqueeze])
 *  C  -->D([MaskedFill])
 *  D  -->I([Squeeze])
 *  I   --> J([ViewCopy])
 *  J   --> K[(out)]
 *
 *  A1[(mask)] -->B1([Contiguous])
 *  B1  -->C1([Cast])
 *  C1  -->D
 *
 *  A2[(value)]-->B2[(Cast)]
 *  B2-->D
 * ```
 *
 * @param [in] selfRef: npu device侧的aclTensor，数据类型支持BOOL、UINT8、INT8、INT16、INT32、INT64、FLOAT、
 *                      FLOAT16、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
 *                      支持非连续的Tensor，数据格式支持ND。
 * @param [in] mask: npu device侧的aclTensor，数据类型支持BOOL。且shape与selfRef满足broadcast关系。数据格式支持ND。
 * @param [in] value: host侧的aclScalar，数据类型需要可转换成selfRef的数据类型。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceMaskedFillScalarGetWorkspaceSize(aclTensor* selfRef, const aclTensor* mask,
                                                                   const aclScalar* value, uint64_t* workspaceSize,
                                                                   aclOpExecutor** executor);
/**
 * @brief aclnnInplaceMaskedFillScalar的第二段接口，用于执行计算。
 *
 * 算子功能：用value填充selfRef里面与mask矩阵中值为true的位置相对应的元素
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *  A[(selfRef)] -->B([Contiguous])
 *  B  -->C([Unsqueeze])
 *  C  -->D([MaskedFill])
 *  D  -->I([Squeeze])
 *  I   --> J([ViewCopy])
 *  J   --> K[(out)]
 *
 *  A1[(mask)] -->B1([Contiguous])
 *  B1  -->C1([Cast])
 *  C1  -->D
 *
 *  A2[(value)]-->B2[(Cast)]
 *  B2-->D
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceMaskedFillScalarGetWorkspaceSize。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceMaskedFillScalar(void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MASKEF_FILL_SCALAR_H_