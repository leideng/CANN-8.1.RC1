/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/license/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_ADDR_H_
#define OP_API_INC_LEVEL2_ACLNN_ADDR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAddr的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：返回输入Tensor中每个元素绝对值的结果
 * 计算公式：
 * $$ out_{i} = addr(self, vec1, vec2, beta=1, alpha=1) $$
 * @param [in] self: addr入参，外积扩展矩阵。npu device侧的aclTensor,
 * 数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL，数据格式支持ND，支持非连续的Tensor。
 * @param [in] vec1: addr入参，外积入参第一向量，npu device侧的aclTensor,
 * 数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL，数据格式支持ND，支持非连续的Tensor。
 * @param [in] vec2: addr入参，外积入参第二向量，npu device侧的aclTensor,
 * 数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL，数据格式支持ND，支持非连续的Tensor。
 * @param [in] betaOptional: addr可选入参，外积扩展矩阵的比例因子，host侧的aclScalar
 * @param [in] alphaOptional: addr可选入参，外积的比例因子，host侧的aclScalar
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包括算子计算流程
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAddrGetWorkspaceSize(const aclTensor* self, const aclTensor* vec1, const aclTensor* vec2,
                                                const aclScalar* betaOptional, const aclScalar* alphaOptional,
                                                aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAddr的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddrGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAddr(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                const aclrtStream stream);

/**
 * @brief aclnnInplaceAddr的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnInplaceAddrGetWorkspaceSize(aclTensor* selfRef, const aclTensor* vec1, const aclTensor* vec2,
                                                       const aclScalar* betaOptional, const aclScalar* alphaOptional,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor);
ACLNN_API aclnnStatus aclnnInplaceAddr(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ADDR_H_