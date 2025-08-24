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
#ifndef OP_API_INC_CONSTANT_PAD_ND_H_
#define OP_API_INC_CONSTANT_PAD_ND_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnConstantPadNd的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：使用输入边界填充输入tensor。
 * @param [in] self: 数据类型支持FLOAT16, FLOAT32, DOUBLE, INT8, INT16, INT32, INT64, UINT8,
 * COMPLEX64, COMPLEX128，支持非连续的Tensor，数据格式支持ND。
 * @param [in] pad: 数据类型支持UINT8、INT8、INT16、INT32、INT64，数组长度必须为偶数且不能超过self维度的两倍。
 * @param [in] out: 数据类型、数据格式与self一致，支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConstantPadNdGetWorkspaceSize(const aclTensor* self, const aclIntArray* pad,
                                                         const aclScalar* value, aclTensor* out,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnConstantPadNd的第二段接口，用于执行计算
 *
 * 算子功能： 使用输入边界填充输入tensor。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnConstantPadNdGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConstantPadNd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CONSTANT_PAD_ND_H_
