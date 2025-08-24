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
#ifndef OP_API_INC_AVGPOOL3D_BACKWARD_H_
#define OP_API_INC_AVGPOOL3D_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAvgPool3dBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：完成avgpool3d的反向计算
 *
 * @param [in] gradOutput: npu
 * device侧的aclTensor，数据类型支持FLOAT, BFLOAT16, FLOAT16。支持4维或者5维。支持非连续的Tensor。支持数据格式为ND。
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT, BFLOAT16, FLOAT16。支持4维或者5维。支持非连续的Tensor。支持数据格式为ND。
 * @param [in] kernelSize: npu
 * device侧的aclIntArray，长度为1(kD=kH=kW)或3(kD,kH,kW)，表示池化窗口大小。数据类型支持INT32和INT64。数值必须大于0。
 * @param [in] stride: npu
 * device侧的aclIntArray，长度为0(默认为kernelSize)或1(sD=sH=sW)或3(sD,sH,sW)，表示池化操作的步长。
 * 数据类型支持INT32和INT64。数值必须大于0。
 * @param [in] padding: npu
 * device侧的aclIntArray，长度为1(padD=padH=padW)或3(padD,padH,padW)，表示在输入的D、H、W方向上padding补0的层数。
 * 数据类型支持INT32和INT64。数值在[0, kernelSize/2]的范围内。
 * @param [in] ceilMode: 数据类型支持BOOL。表示计算输出shape时，向下取整（False），否则向上取整。
 * @param [in] countIncludePad: 数据类型支持BOOL。表示平均计算中包括零填充（True），否则不包括。
 * @param [in] divisorOverride: 数据类型支持INT64。如果指定，它将用作平均计算中的除数，当值为0时，该属性不生效。
 * @param [out] output: npu
 * device侧的aclTensor，输出Tensor，数据类型支持FLOAT16、BFLOAT16和FLOAT。支持4维或5维。支持非连续的Tensor。支持数据格式为ND。
 * 数据类型、数据格式需要与gradOutput一致。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAvgPool3dBackwardGetWorkspaceSize(const aclTensor* gradOuput, const aclTensor* self,
                                                             const aclIntArray* kernelSize, const aclIntArray* stride,
                                                             const aclIntArray* padding, bool ceilMode, bool countIncludePad,
                                                             int64_t divisorOverride, aclTensor* output,
                                                             uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAvgPool3dBackward的第二段接口，用于执行计算。
 *
 * 算子功能：完成avgpool3d的反向计算
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAvgPool3dBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAvgPool3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_AVGPOOL3D_BACKWARD_H_
