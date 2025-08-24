/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at **
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_GELUV2_H_
#define OP_API_INC_LEVEL2_ACLNN_GELUV2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnGeluV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：高斯误差线性单元激活函数
 * 计算公式：
 * $$ y_{i}=Gelu(x_{i})=x_{i}×Φ(x_{i}) $$
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *     A[(x)]  --> B{l0op::Contiguous}
 *     B -->C([l0op::GeluV2])
 *     C --> D{l0op::ViewCopy}
 *     D --> E[(y)]
 *     F[approximate（可选）]-->G[getApproximateStr]-->C
 * ```
 *
 * @param [in] x: 待进行gelu_v2计算的入参。npu device侧的aclTensor。
 * 数据类型支持FLOAT16、FLOAT32、BFLOAT16，且数据类型必须和y一样，数据格式支持ND，shape必须和y一样，支持非连续的Tensor。
 * @param [in] approximate: gelu_v2计算的入参，指定高斯近似算法，默认值为0（表示：0: "none", 1: "tanh" ）。
 * @param [in] y: gelu_v2计算的出参。
 * npu
 * device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16，且数据类型必须和x一样，数据格式支持ND，shape必须和x一样，
 * 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGeluV2GetWorkspaceSize(const aclTensor* x, int64_t approximate, aclTensor* y,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnGeluV2的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGeluv2GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGeluV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GELUV2_H_