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

#ifndef OP_API_INC_UNAMPLE_NEAREST_2D_BACKWARD_H_
#define OP_API_INC_UNAMPLE_NEAREST_2D_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleNearest2dBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：upsample_nearest2d的反向计算。
 *
 * @param [in] gradOut: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16。
 * 支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持NCHW和NHWC（[参考](#参考)）。
 * @param [in] outputSize: 输入IntArray，size大小为2。表示输入gradOut在H和W维度上的空间大小。
 * @param [in] inputSize: 输入IntArray，size大小为4。表示输出out分别在N、C、H和W维度上的空间大小。
 * @param [in] scalesH: double常量，表示输出out的height维度乘数。
 * @param [in] scalesW: double常量，表示输出out的width维度乘数。
 * @param [out] gradInput: npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16，且数据类型与gradOut的数据类型一致。
 * 支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持NCHW和NHWC（[参考](#参考)）。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUpsampleNearest2dBackwardGetWorkspaceSize(const aclTensor* gradOut,
                                                                     const aclIntArray* outputSize,
                                                                     const aclIntArray* inputSize, double scalesH,
                                                                     double scalesW, aclTensor* gradInput,
                                                                     uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnUpsampleNearest2dBackward的第二段接口，用于执行计算。
 *
 * 算子功能：upsample_nearest2d的反向计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，
 * 由第一段接口aclnnUpsampleNearest2dBackwardGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUpsampleNearest2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_NEAREST_2D_BACKWARD_H_
