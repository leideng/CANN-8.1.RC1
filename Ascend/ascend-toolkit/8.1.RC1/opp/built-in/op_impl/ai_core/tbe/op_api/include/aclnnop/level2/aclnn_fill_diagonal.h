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
#ifndef OP_API_INC_ADD_H_
#define OP_API_INC_ADD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInplaceFillDiagonal的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：以fillValue填充tensor对角线
 *
 * @param [in] selfRef: npu device侧的aclTensor，输入selfRef，数据类型支持FLOAT、FLOAT16、INT32、INT64。
 * 支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND（[参考](#参考)）。
 * @param [in] fillValue:
 * host侧的aclScalar，输入fillValue，数据类型支持FLOAT、FLOAT16、DOUBLE、UINT8、INT8、INT16、INT32 INT64、BOOL。
 * @param [in] wrap: 输入wrap，数据类型支持BOOL，是否每经过`min(col, row)`行形成一条新的对角线，仅对二维输入Tensor有效。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceFillDiagonalGetWorkspaceSize(aclTensor* selfRef, const aclScalar* fillValue,
                                                               bool wrap, uint64_t* workspaceSize,
                                                               aclOpExecutor** executor);

/**
 * @brief aclnnInplaceFillDiagonal的第二段接口，用于执行计算。
 *
 * 算子功能：以fillValue填充tensor对角线
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceFillDiagonalGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceFillDiagonal(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ADD_H_
