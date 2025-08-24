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
#ifndef OP_API_INC_LEVEL2_ACLNN_LERP_SCALAR_H_
#define OP_API_INC_LEVEL2_ACLNN_LERP_SCALAR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLerps的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：根据给定的权重，在起始和结束Tensor之间进行线性插值，返回插值后的Tensor。

 *
 * 计算图：
 * graph LR
 * ```mermaid
 * A1[(self)] --> B1([l0op::Contiguous])
 * B1 --> C([l0op::Lerp])
 * A2[(end)] --> B2([l0op::Contiguous])
 * B2 --> C
 * A3((weight)) --> C
 * C --> D([l0op::Cast])
 * D --> E([l0op::ViewCopy]) --> F[(out)]
 * ```
 *
 * @param [in] self: 公式中的输入start，数据类型支持FLOAT16、FLOAT，shape需要与end满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] end: 公式中的输入end，数据类型支持FLOAT16、FLOAT且与self的数据类型一致，shape需要与self满足broadcast关系。
 * 支持非连续Tensor，数据格式支持ND。
 * @param [in] weight: 公式中的输入weight，数据类型支持FLOAT16、FLOAT。
 * @param [in] out: 公式中的out，数据类型支持FLOAT16、FLOAT且与self的数据类型一致、shape与self和end
 * broadcast之后的shape一致。支持非连续Tensor，数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLerpsGetWorkspaceSize(const aclTensor* self, const aclTensor* end, const aclScalar* weight,
                                                 aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLerps的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnLerpsGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLerps(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                 const aclrtStream stream);

/**
 * @brief aclnnInplaceLerps的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：根据给定的权重，在起始和结束Tensor之间进行线性插值，返回插值后的Tensor。

 *
 * @param [in] selfRef: 公式中的输入start，数据类型支持FLOAT16、FLOAT，shape需要与end满足broadcast关系，且broadcast后的shape与selfRef一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] end: 公式中的输入end，数据类型支持FLOAT16、FLOAT且与self的数据类型一致，shape需要与selfRef满足broadcast关系，且broadcast后的shape与selfRef一致。
 * 支持非连续Tensor，数据格式支持ND。
 * @param [in] weight: 公式中的输入weight，数据类型支持FLOAT16、FLOAT。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceLerpsGetWorkspaceSize(aclTensor* selfRef, const aclTensor* end,
                                                        const aclScalar* weight, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor);

/**
 * @brief aclnnInplaceLerps的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceLerpsGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceLerps(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_LERP_SCALAR_H_