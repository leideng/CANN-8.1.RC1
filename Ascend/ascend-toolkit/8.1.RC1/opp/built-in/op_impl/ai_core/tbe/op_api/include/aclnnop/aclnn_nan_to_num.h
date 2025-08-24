/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_NANTONUM_H_
#define OP_API_INC_LEVEL2_ACLNN_NANTONUM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnNand的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：将输入中的NaN、正无穷大和负无穷大分别替换为nan、posinf和neginf指定的值。
 *
 * @param [in] self: 待进行NanToNum计算的入参。npu device侧的aclTensor，
 * 数据类型支持FLOAT32、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL，
 * 数据格式支持ND，且数据格式需要与out一致， 支持非连续的Tensor。
 * @param [in] nan: 输入参数。数据类型支持FLOAT，替换tensor元素中NaN的值。
 * @param [in] posinf: 输入参数。数据类型支持FLOAT，替换tensor元素中正无穷大的值。
 * @param [in] neginf: 输入参数。数据类型支持FLOAT，替换tensor元素中负无穷大的值。
 * @param [out] out: NanToNum计算的出参。npu device侧的aclTensor，
 * 数据类型支持FLOAT32、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL，
 * 数据格式支持ND，且数据格式需要与self一致， 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNanToNumGetWorkspaceSize(const aclTensor* self, float nan, float posinf, float neginf,
                                                    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnNanToNum的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnNanToNumGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNanToNum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

/**
 * @brief aclnnInplaceNanToNum的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] selfRef: 待进行NanToNum计算的入参。npu device侧的aclTensor，
 * 数据类型支持FLOAT64、FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND，且数据格式需要与out一致， 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceNanToNumGetWorkspaceSize(aclTensor* selfRef, float nan, float posinf, float neginf,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceNanToNum的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnNanToNumGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceNanToNum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_NANTONUM_H_