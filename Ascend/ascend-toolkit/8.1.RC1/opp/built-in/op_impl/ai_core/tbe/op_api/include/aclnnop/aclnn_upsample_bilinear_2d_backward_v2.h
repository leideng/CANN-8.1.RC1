/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef OP_API_INC_UNAMPLE_BILINEAR_2D_BACKWARD_V2_H_
#define OP_API_INC_UNAMPLE_BILINEAR_2D_BACKWARD_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleBilinear2dBackwardV2的第二段接口，用于执行计算。
 * 
 * 算子功能：aclnnUpsampleBilinear2d的反向传播。
 *
 * @param [in] workspace: 在Device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在Device侧申请的workspace大小。
 * 由第一段接口aclnnUpsampleBilinear2dBackwardV2GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的AscendCL Stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUpsampleBilinear2dBackwardV2(void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
                                                      aclrtStream stream);

/**
 * @brief aclnnUpsampleBilinear2dBackwardV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * 
 * 算子功能：aclnnUpsampleBilinear2d的反向传播。
 *
 * @param [in] gradOutput: Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16。
 * 支持非连续的Tensor，数据格式支持NCHW，shape仅支持四维Tensor。数据类型与出参out的数据类型一致。
 * @param [in] outputSize: Host侧的aclIntArray，数据类型支持INT64，size大小为2。表示输入gradOutput在H和W维度上的空间大小。
 * @param [in] inputSize: Host侧的aclIntArray，数据类型支持INT64，size大小为4。表示输出out分别在N、C、H和W维度上的空间大小。
 * @param [in] scalesH: Host侧的浮点型，表示输出out的height维度乘数。
 * @param [in] scalesW: Host侧的浮点型，表示输出out的width维度乘数。
 * @param [out] out: Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16。
 * 支持非连续的Tensor，数据格式支持NCHW，shape仅支持四维Tensor。数据类型与入参gradOutput的数据类型一致。
 * @param [out] workspaceSize: 返回用户需要在Device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUpsampleBilinear2dBackwardV2GetWorkspaceSize(
    const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize, bool alignCorners,
    double scalesH, double scalesW, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_BILINEAR_2D_BACKWARD_V2_H_