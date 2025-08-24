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
#ifndef OP_API_INC_SCATTER_ND_UPDATE_H_
#define OP_API_INC_SCATTER_ND_UPDATE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnScatterNdUpdate的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnnop_ops_infer
 * @domain aclnnop_ops_train
 * 算子功能： 将tensor updates中的值按指定的索引indices逐个更新tensor var中的值。
 * @param [in] varRef: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, BOOL
 * INT64，BFLOAT16，支持非连续的Tensor，数据格式支持ND。
 * @param [in] indices: npu device侧的aclTensor，数据类型支持INT32, INT64类型。支持非连续的Tensor，数据格式支持ND。
 * @param [in] updates: npu device侧的aclTensor，数据类型支持FLOAT16, FLOAT32, BOOL
 * INT64，BFLOAT16，支持非连续的Tensor，数据格式支持ND,
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnScatterNdUpdateGetWorkspaceSize(aclTensor* varRef, const aclTensor* indices,
                                                           const aclTensor* updates, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor);

/**
 * @brief: aclnnScatterNdUpdate的第二段接口，用于执行计算
 * @domain aclnnop_ops_infer
 * @domain aclnnop_ops_train
 * 算子功能： 将tensor updates中的值按指定的索引indices逐个更新tensor var中的值。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnScatterNdUpdateGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnScatterNdUpdate(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SCATTER_ND_UPDATE_H_