/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef OP_API_INC_LEVEL2_ACLNN_GROUP_QUANT_H_
#define OP_API_INC_LEVEL2_ACLNN_GROUP_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: 待进行GroupQuant计算的入参。npu device侧的aclTensor，
 * 数据类型支持float16, bfloat16, float32, 数据格式支持ND，
 * 支持非连续的Tensor。
 * @param [in] scale: npu device侧的aclTensor, 数据类型支持float, bf16, float16
 * @param [in] groupIndex: npu device侧的aclTensor，数据类型支持float, bf16, float16
 * @param [in] offsetOptional:  npu device侧的aclTensor，数据类型int, int64
 * @param [in] dstType:  host侧的aclScalar, 数据类型int
 * @param [in] y: GroupQuant计算的出参。npu device侧的aclTensor，
 * 数据类型支持int8, int4, 数据格式支持ND，
 * 支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* scale,
                                                      const aclTensor* groupIndex, const aclTensor* offsetOptional,
                                                      int32_t dstType, aclTensor* y,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGroupQuant的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnGroupQuantGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupQuant(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GROUP_QUANT_H_