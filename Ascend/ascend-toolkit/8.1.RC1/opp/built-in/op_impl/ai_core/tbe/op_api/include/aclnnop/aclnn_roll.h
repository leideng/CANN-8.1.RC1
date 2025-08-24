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
#ifndef OP_API_ACLNN_ROLL_H
#define OP_API_ACLNN_ROLL_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRoll的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor完成roll操作
 * @param [in] x: device侧的aclTensor，数据类型支持BFLOAT16,FLOAT16, FLOAT32, INT8, UINT8, INT32, UINT32，BOOL。支持
 * [非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [in] shifts: int64的数组，数组长度与dims保持一致。
 * @param [in] dims: int64的数组，数组长度与shifts保持一致，取值范围在[-x.dim(), x.dim() -
 * 1]之内，例如：x的维度是4，则取值范 围在[-4, 3]。
 * @param [in] out: device侧的aclTensor，数据类型和数据格式与输入x保持一致
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRollGetWorkspaceSize(const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims,
                                                aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnRoll的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成roll操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnRollGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRoll(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_ACLNN_ROLL_H