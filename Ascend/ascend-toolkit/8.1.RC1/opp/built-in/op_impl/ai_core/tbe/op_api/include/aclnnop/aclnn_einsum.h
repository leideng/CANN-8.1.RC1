/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifndef OP_API_INC_EINSUM_H_
#define OP_API_INC_EINSUM_H_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnEinsum的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：使用爱因斯坦求和约定对张量序列计算代数张量运算
 * @domain aclnn_ops_infer
 * @param [in]   tensors:
 * 输入TensorList，数据类型支持FLOAT16、FLOAT、INT16、UINT16、INT32、UINT32、INT64、UINT64。支持非连续Tensor，数据格式支持ND。
 * @param [in]   equation: 输入字符串，数据类型支持const char *。
 * @param [out]  output:
 * 输出Tensor，数据类型支持FLOAT16、FLOAT、INT16、UINT16、INT32、UINT32、INT64、UINT64。支持非连续Tensor，数据格式支持ND。
 * @param [out]  workspaceSize：返回用户需要在npu device侧申的的workspace大小。
 * @param [out]  executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnEinsumGetWorkspaceSize(const aclTensorList* tensors, const char* equation, aclTensor* output,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnEinsum的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAddGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEinsum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EINSUM_H_