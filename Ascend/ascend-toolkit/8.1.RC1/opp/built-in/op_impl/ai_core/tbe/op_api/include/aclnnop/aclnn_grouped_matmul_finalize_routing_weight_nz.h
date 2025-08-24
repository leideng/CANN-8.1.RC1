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
#ifndef OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_NZ
#define OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_NZ

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulFinalizeRoutingWeightNz的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：实现GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作, 当前仅支持w为nz格式。
 * @param [in] x: matmul左矩阵，数据类型支持：int8。
 * @param [in] w: matmul右矩阵，数据类型支持：int8。
 * @param [in] scaleOptional: 量化参数中的缩放因子，数据类型支持：float32。
 * @param [in] biasOptional: 偏置，数据类型支持：float32。
 * @param [in] pertokenScaleOptional: 反量化参数，数据类型支持：float32。
 * @param [in] groupListOptional: 代表输入和输出分组轴方向的matmul大小分布，数据类型支持：int64。
 * @param [in] sharedInputOptional: 
 * moe计算中共享专家的输出，需要与moe专家的输出进行combine操作，数据类型支持：bfloat16、float16。
 * @param [in] logitOptional: 
 * moe专家对各个token的logit大小，矩阵乘的计算输出与该logit做乘法，然后索引进行combine，数据类型支持：float32。
 * @param [in] rowIndexOptional: 
 * moe专家输出按照该rowIndex进行combine，其中的值即为combine做scatter add的索引，数据类型支持：int64、int32。
 * @param [in] sharedInputWeight: 
 * 共享专家与moe专家进行combine的系数，shareInput先于该参数乘，然后在和moe专家结果累加，数据类型支持：float32。
 * @param [in] sharedInputOffset: 共享专家输出的在总输出中的偏移，数据类型支持：int64。
 * @param [in] transposeX: 左矩阵是否转置，默认值：false。
 * @param [in] transposeW: 右矩阵是否转置，默认值：false。
 * @param [in] groupListType: GroupedMatmul分组类型，默认值：1，count模式，数据类型支持：int32。。
 * @param [out] y: 计算结果，数据类型：float32，float16, bfloat16。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *w, const aclTensor *scaleOptional,
                                                     const aclTensor* biasOptional, const aclTensor *pertokenScaleOptional, 
                                                     const aclTensor *groupListOptional, const aclTensor *sharedInputOptional, const aclTensor* logitOptional,
                                                     const aclTensor *rowIndexOptional, int64_t dtype, float sharedInputWeight,  
                                                     int64_t sharedInputOffset, bool transposeX, bool transposeW, int64_t groupListType, aclTensor *y,
                                                     uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulFinalizeRoutingWeightNz的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulFinalizeRoutingNZGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_MATMUL_NZ