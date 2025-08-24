/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/license/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_H_
#define OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_H_

#include <cstring>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRoiAlign的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：ROIAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32，支持非连续的Tensor，数据格式支持NCHW。
 * @param [in] rois: npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32，支持非连续的Tensor，数据格式支持ND。
 * @param [in] batchIndices: npu device侧的aclTensor，数据类型支持INT64，支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT16、FLOAT32，和self一致，支持非连续的Tensor，数据格式支持NCHW。
 * @param [in] mode: host侧的string类型，池化模式，支持"avg"或"max"。
 * @param [in] outputHeight: host侧的int类型，ROI输出特征图的H。
 * @param [in] outputWidth: host侧的int类型，ROI输出特征图的W。
 * @param [in] samplingRatio: host侧的int类型，用于计算每个输出元素的和W上的bin数。
 * @param [in] spatialScale: host侧的float类型，缩放因子，用于将ROI坐标转换为输入特征图。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRoiAlignGetWorkspaceSize(const aclTensor* self, const aclTensor* rois,
                                                    const aclTensor* batchIndices, const char* mode, int outputHeight,
                                                    int outputWidth, int samplingRatio, float spatialScale,
                                                    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnRoiAlign的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnRoiAlignGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRoiAlign(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_H_