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

#ifndef OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_V2_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_V2_BACKWARD_H_

#include <cstring>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRoiAlignV2Backward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：aclnnRoiAlignV2Backward是aclnnRoiAlign的反向传播。ROIAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。
 *
 * @param [in] gradOutput: npu device侧的aclTensor，数据类型支持FLOAT32，支持非连续的Tensor，数据格式支持NCHW。
 * @param [in] boxes: npu device侧的aclTensor，数据类型支持FLOAT32，支持非连续的Tensor，数据格式支持ND。
 * @param [in] inputShape: aclIntArray，指定输出的shape。
 * @param [in] pooledHeight: host侧的int类型，正向RoiAlign池化后输出图像的H。
 * @param [in] pooledWidth: host侧的int类型，正向RoiAlign池化后输出图像的W。
 * @param [in] spatialScale: host侧的float类型，缩放因子，用于将ROI坐标转换为输入特征图。
 * @param [in] samplingRatio: host侧的int类型，用于计算每个输出元素的和W上的bin数。
 * @param [in] aligned: host侧的bool类型，用于判断像素框是否偏移-0.5来更好对齐。
 * @param [out] gradInput: npu
 * device侧的aclTensor，数据类型支持FLOAT32，和gradOutput一致，支持非连续的Tensor，数据格式支持NCHW。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRoiAlignV2BackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* boxes,
                        const aclIntArray* inputShape, int64_t pooledHeight, int64_t pooledWidth, 
                        float spatialScale, int64_t samplingRatio, bool aligned, aclTensor* gradInput, 
                        uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnRoiAlignV2Backward的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnRoiAlignV2BackwardGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRoiAlignV2Backward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_BACKWARD_H_