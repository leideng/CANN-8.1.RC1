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

#ifndef OP_API_INC_PROD_H_
#define OP_API_INC_PROD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnProd的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：返回输入tensor的元素的乘积。
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BOOL。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] dtype: host侧的aclDataType，输出tensor的数据类型，需要与out的数据类型一致。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BOOL、COMPLEX64、
 * COMPLEX128，且数据类型与dtype一致。支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnProdGetWorkspaceSize(const aclTensor* self, const aclDataType dtype, aclTensor* out,
                                                uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnProd的第二段接口，用于执行计算。
 *
 * 算子功能：返回输入tensor的元素的乘积。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnProdGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnProd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnProdDim的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：返回输入tensor给定维度上每行的乘积。
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BOOL。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] dim: host侧的int64，指定要缩减的维度。
 * @param [in] keepDim: host侧的bool，输出tensor是否保留维度。
 * @param [in] dtype: host侧的aclDataType，输出tensor的数据类型，需要与out的数据类型一致。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BOOL、COMPLEX64、
 * COMPLEX128，且数据类型与dtype一致。支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnProdDimGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepDim,
                                                   const aclDataType dtype, aclTensor* out, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);
/**
 * @brief aclnnProdDim的第二段接口，用于执行计算。
 *
 * 算子功能：返回输入tensor给定维度上每行的乘积。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnProdDimGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnProdDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_PROD_H_
