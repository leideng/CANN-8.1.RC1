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
#ifndef OP_API_INC_GRID_SAMPLER2D_BACKWARD_H_
#define OP_API_INC_GRID_SAMPLER2D_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGridSampler2DBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：aclnnGridSampler2D的反向计算。
 *
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(gradOutput)] --> B([l0op::Contiguous]) --> C([l0op::GridSampler2DGrad])
 *     D[(input)] --> E([l0op::Contiguous]) --> C
 *     F[(grid)] --> G([l0op::Contiguous]) --> C
 *     H((interpolationMode)) --> C
 *     I((paddingMode)) --> C
 *     J((alignCorners)) --> C
 *     C --> K([l0op::ViewCopy]) --> Out1[("inputGrad")]
 *     C --> L([l0op::ViewCopy]) --> Out2[("gridGrad")]
 * ```
 *
 * @param [in] gradOutput: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE，支持非连续的Tensor，数据格式支持ND。
 * @param [in] input: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE，支持非连续的Tensor，数据格式支持ND。
 * @param [in] grid: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE，支持非连续的Tensor，数据格式支持ND。
 * @param [in] interpolationMode：int64_t类型，表示插值模式，0：bilinear（双线性插值），1：nearest（最邻近插值）。
 * @param [in] paddingMode：int64_t类型，表示填充模式，即当（x,y）取值超过输入特征图采样范围时，返回一个特定值，
 * 有0：zeros、1：border、2：reflection三种模式。
 * @param [in] alignCorners：bool类型，表示设定特征图坐标与特征值的对应方式，设定为true时，特征值位于像素中心。
 * @param [in] outputMask：aclBoolArray类型，输出的掩码。
 * @param [in] inputGrad: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE，支持非连续的Tensor，数据格式支持ND。
 * @param [in] gridGrad: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGridSampler2DBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* input,
                                                                 const aclTensor* grid, int64_t interpolationMode,
                                                                 int64_t paddingMode, bool alignCorners,
                                                                 const aclBoolArray* outputMask, aclTensor* inputGrad,
                                                                 aclTensor* gridGrad, uint64_t* workspaceSize,
                                                                 aclOpExecutor** executor);

/**
 * @brief aclnnGridSampler2DBackward的第二段接口，用于执行计算。
 *
 * 算子功能：aclnnGridSampler2D的反向计算。
 *
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(gradOutput)] --> B([l0op::Contiguous]) --> C([l0op::GridSampler2DGrad])
 *     D[(input)] --> E([l0op::Contiguous]) --> C
 *     F[(grid)] --> G([l0op::Contiguous]) --> C
 *     H((interpolationMode)) --> C
 *     I((paddingMode)) --> C
 *     J((alignCorners)) --> C
 *     C --> K([l0op::ViewCopy]) --> Out1[("inputGrad")]
 *     C --> L([l0op::ViewCopy]) --> Out2[("gridGrad")]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGridSampler2DBackwardGetWorkspaceSize
 * 获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGridSampler2DBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GRID_SAMPLER2D_BACKWARD_H_