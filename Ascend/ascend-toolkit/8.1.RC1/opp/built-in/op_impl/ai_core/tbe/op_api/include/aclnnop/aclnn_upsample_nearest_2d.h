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

#ifndef OP_API_INC_UNAMPLE_NEAREST_2D_H_
#define OP_API_INC_UNAMPLE_NEAREST_2D_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleNearest2D的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *  A[(self)] -.->B([l0op::Contiguous])
 *  B --> D([l0op::TransData])
 *  D --> I([l0op::ResizeNearestNeighborV2])
 *  C[(outputSize)] --> I
 *  I --> J([l0op::TransData])
 *  J -.-> P([l0op::ViewCopy])
 *  P --> G[(out)]
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、DOUBLE、UIN8。数据格式支持NCHW、NHWC。支持非连续的Tensor。
 * @param [in] outputSize: npu device侧的aclIntArray，指定输出空间大小。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、DOUBLE、UIN8。数据格式支持NCHW、NHWC。支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUpsampleNearest2dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize,
                                                             aclTensor* out, uint64_t* workspaceSize,
                                                             aclOpExecutor** executor);

/**
 * @brief aclnnUpsampleNearest2D的第二段接口，用于执行计算。
 *
 * 算子功能：对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *  A[(self)] -.->B([l0op::Contiguous])
 *  B --> D([l0op::TransData])
 *  D --> I([l0op::ResizeNearestNeighborV2])
 *  C[(outputSize)] --> I
 *  I --> J([l0op::TransData])
 *  J -.-> P([l0op::ViewCopy])
 *  P --> G[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnUpsampleNearest2dGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUpsampleNearest2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_NEAREST_2D_H_
