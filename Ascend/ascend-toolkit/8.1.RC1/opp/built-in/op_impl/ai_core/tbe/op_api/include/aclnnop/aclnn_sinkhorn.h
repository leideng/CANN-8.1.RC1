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
#ifndef OP_API_INC_SINKHORN_H_
#define OP_API_INC_SINKHORN_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSinkhorn的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：计算Sinkhorn距离
 *
 * @param [in] cost: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，数据格式支持ND.
 * @param [in] tol: 误差，支持FLOAT类型，如果传入空指针，则tol取0.0001。
 * @param [in] p: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，数据格式支持ND.
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSinkhornGetWorkspaceSize(
    const aclTensor* cost, 
    const aclScalar* tol,
    aclTensor* p,
    uint64_t* workspaceSize, 
    aclOpExecutor** executor);
/**
 * @brief aclnnSinkhorn的第二段接口，用于执行计算。
 *
 * 算子功能：对输入的 Tensor cost， 计算Sinkhorn距离，形成一个新的 Tensor，并返回。 
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSinkhornGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSinkhorn(
    void* workspace, 
    uint64_t workspaceSize, 
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SINKHORN_H_
