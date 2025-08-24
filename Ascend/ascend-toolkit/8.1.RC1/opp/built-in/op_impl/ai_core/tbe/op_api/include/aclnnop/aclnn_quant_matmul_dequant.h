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
#ifndef OP_API_INC_QUANT_MATMUL_DEQUANT_H_
#define OP_API_INC_QUANT_MATMUL_DEQUANT_H_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantMatmulDequant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 功能描述：对输入 x 进行量化，矩阵乘以及反量化。
 * @param [in] x: npu device侧的aclTensor，支持非连续的Tensor，数据类型支持FLOAT16，数据格式支持ND。
 * @param [in] weight: npu device侧的aclTensor，数据类型支持INT8，数据格式支持ND和NZ。
 * @param [in] weightScale: npu device侧的aclTensor，支持非连续的Tensor，数据类型支持FLOAT，数据格式支持ND。
 * @param [in] biasOptional: npu device侧的aclTensor，支持非连续的Tensor，数据类型支持INT32，数据格式支持ND。
 * @param [in] xScaleOptional: npu device侧的aclTensor，支持非连续的Tensor，数据类型支持FLOAT，数据格式支持ND。
 * @param [in] xOffsetOptional: npu device侧的aclTensor，支持非连续的Tensor，数据类型支持FLOAT，数据格式支持ND。
 * @param [in] smoothScaleOptional: npu device侧的aclTensor，支持非连续的Tensor，数据类型支持FLOAT16，数据格式支持ND。
 * @param [in] xQuantMode: host侧的char*类型，表示x的量化模式。
 * @param [in] transposeWeight: host侧的bool类型，表示weight是否转置。
 * @param [out] out: npu device侧的aclTensor，只支持连续Tensor，数据类型支持FLOAT16，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnQuantMatmulDequantGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, 
                                                    const aclTensor *weightScale, const aclTensor *biasOptional, 
                                                    const aclTensor *xScaleOptional, const aclTensor *xOffsetOptional, 
                                                    const aclTensor *smoothScaleOptional, 
                                                    char *xQuantMode, bool transposeWeight, const aclTensor *out, 
                                                    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnQuantMatmulDequant的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnQuantMatmulDequantGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnQuantMatmulDequant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_MATMUL_DEQUANT_H_