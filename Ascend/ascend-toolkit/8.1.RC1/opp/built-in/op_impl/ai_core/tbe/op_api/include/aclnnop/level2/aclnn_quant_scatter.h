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
#ifndef OP_API_INC_LEVEL2_ACLNN_QUANT_SCATTER_H_
#define OP_API_INC_LEVEL2_ACLNN_QUANT_SCATTER_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInplaceQuantScatter的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：
 * 先将updates进行量化，然后将updates中的值按指定的轴axis和索引indices逐个更新selfRef中的值。该算子为自定义算子语义,
 * 无对应的tensorflow或pytorch接口。
 * @param [in] selfRef: npu device侧的aclTensor，数据类型支持INT8。维度数量需要与updates一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] indices: npu device侧的aclTensor，数据类型支持INT32。支持非连续的Tensor，数据格式支持ND。
 * @param [in] updates: npu device侧的aclTensor，数据类型支持BFLOAT16（仅昇腾910B AI处理器支持）。
 * 维度数量需要与selfRef一致。支持非连续的Tensor，数据格式支持ND。
 * @param [in] quantScales: npu device侧的aclTensor，数据类型支持BFLOAT16（仅昇腾910B AI处理器支持）类型。
 * 支持非连续的tensor。数据格式支持ND。
 * @param [in] quantZeroPoints: npu device侧的aclTensor，数据类型支持BFLOAT16（仅昇腾910B AI处理器支持）类型。
 * 支持非连续的tensor。数据格式支持ND。
 * @param [in] axis: 用来scatter的维度，数据类型为INT64。
 * @param [in] quantAxis: 用来量化的维度，数据类型为INT64。
 * @param [in] reduction: 指定要应用到输出的缩减，数据类型为INT64。当前仅支持1('update')。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnInplaceQuantScatterGetWorkspaceSize(aclTensor* selfRef, const aclTensor* indices,
                                                               const aclTensor* updates, const aclTensor* quantScales,
                                                               const aclTensor* quantZeroPoints, int64_t axis,
                                                               int64_t quantAxis, int64_t reduction,
                                                               uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnInplaceQuantScatter的第二段接口，用于执行计算
 * @domain aclnn_ops_infer
 * 算子功能：
 * 先将updates进行量化，然后将updates中的值按指定的轴axis和索引indices逐个更新selfRef中的值。该算子为自定义算子语义,
 * 无对应的tensorflow或pytorch接口。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceQuantScatterGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceQuantScatter(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_QUANT_SCATTER_H_