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

#ifndef OP_API_INC_AFFINE_GRID_H_
#define OP_API_INC_AFFINE_GRID_H_
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAffineGrid的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：给定一组仿射矩阵(theta)，生成一个2D或3D的网络来表示仿射变换后图像的坐标在原图像上的坐标位置。
 *
 * @param [in] theta: npu device侧的aclTensor，表示仿射变换的仿射参数。
 * 数据类型支持FLOAT和FLOAT16。shape是(N, 2, 3)或(N, 3, 4)。支持非连续的Tensor，数据格式支持ND。
 * @param [in] size: host侧的aclIntArray，表示要要输出的图像的size。大小为4(N, C, H, W)或者5(N, C, D, H, W)。
 * @param [in] alignCorners: host侧的bool。
 * @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT和FLOAT16，且数据类型与theta一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAffineGridGetWorkspaceSize(const aclTensor* theta, const aclIntArray* size,
                                                      bool alignCorners, aclTensor* out, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief aclnnAffineGrid的第二段接口，用于执行计算。
 *
 * 算子功能：给定一组仿射矩阵(theta)，生成一个2D或3D的网络来表示仿射变换后图像的坐标在原图像上的坐标位置。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAffineGridGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAffineGrid(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_AFFINE_GRID_H_
