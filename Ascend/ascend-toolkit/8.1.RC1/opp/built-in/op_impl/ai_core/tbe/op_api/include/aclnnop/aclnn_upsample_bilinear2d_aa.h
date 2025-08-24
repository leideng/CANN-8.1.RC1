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

#ifndef OP_API_INC_UPSAMPLE_BILINEAR2D_AA_H_
#define OP_API_INC_UPSAMPLE_BILINEAR2D_AA_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleBilinear2dAA的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：对由多个输入通道组成的输入信号应用2D双线性抗锯齿采样。
 * @domain aclnn_ops_infer
 * 参数描述:
 * @Param [in] input 输入Tensor，数据类型支持FLOAT，FLOAT16，BFLOAT16。支持非连续Tensor，数据格式支持NCHW，shape维度仅支持4维的Tensor。数据类型与出参out的数据类型一致。
 * @Param [in] outputSize 输出空间大小，要求是二维数组，数据类型支持INT64，取值和out的H、W维度一样。
 * @Param [in] alignCorners 是否是角对齐。如果设置为true，则输入和输出张量按其角像素的中心点对齐，保留角像素处的值。如果设置为false，则输入和输出张量通过其角像素的角点对齐，并使用边缘值对边界外的值进行填充。
 * @Param [in] scalesH 空间大小的height维度乘数。
 * @Param [in] scalesW 空间大小的width维度乘数。
 * @Param [in] out 数据类型支持FLOAT、FLOAT16、BFLOAT16。支持非连续的Tensor，数据格式支持NCHW。shape维度仅支持4维的Tensor。数据类型与入参input的数据类型一致。
 * @Param [out] workspaceSize 返回用户需要在npu device侧申请的workspace大小。
 * @Param [out] executor 返回op执行器，包含了算子计算流程。
 */
ACLNN_API aclnnStatus aclnnUpsampleBilinear2dAAGetWorkspaceSize(const aclTensor* input, const aclIntArray* outputSize,
                                                                bool alignCorners, double scalesH,
                                                                double scalesW, aclTensor* out,
                                                                uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnUpsampleBilinear2dAA的第二段接口，用于执行计算。
 * 功能描述：对由多个输入通道组成的输入信号应用2D双线性抗锯齿采样。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnUpsampleBilinear2dAA(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UPSAMPLE_BILINEAR2D_AA_H_