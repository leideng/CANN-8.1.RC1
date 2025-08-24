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
#ifndef OP_API_INC_MASKED_SCATTER_H_
#define OP_API_INC_MASKED_SCATTER_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMaskedScatter的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：根据掩码mask张量中元素为True的位置，复制source中的元素到selfRef对应的位置上。
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *   A[(selfRef)] -->B([l0op::Contiguous])
 *   B  -->D([l0op::MaskedScatter])
 *   D  -->J([l0op::ViewCopy])
 *   J   --> K[(selfRef)]
 *   A2[(mask)] -->B2([l0op::Contiguous])
 *   B2 --> C2([l0op::Cast])
 *   C2  -->D
 *   A1[(source)]-->B1([l0op::Contiguous])
 *   B1-->D
 * ```
 *
 * @param [in] selfRef: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] mask: npu device侧的aclTensor，数据类型支持BOOL、UINT8。shape不能大于selfRef，且需要和selfRef满足
 * broadcast关系。数据格式支持ND。
 * @param [in] source:npu device侧的aclTensor，数据类型需要与selfRef相同。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceMaskedScatterGetWorkspaceSize(aclTensor* selfRef, const aclTensor* mask,
                                                                const aclTensor* source, uint64_t* workspaceSize,
                                                                aclOpExecutor** executor);
/**
 * @brief aclnnMaskedScatter的第二段接口，用于执行计算。
 *
 * 算子功能：根据掩码mask张量中元素为True的位置，复制source中的元素到self对应的位置上。
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *   A[(selfRef)] -->B([l0op::Contiguous])
 *   B  -->D([l0op::MaskedScatter])
 *   D  -->J([l0op::ViewCopy])
 *   J   --> K[(selfRef)]
 *   A2[(mask)] -->B2([l0op::Contiguous])
 *   B2 --> C2([l0op::Cast])
 *   C2  -->D
 *   A1[(source)]-->B1([l0op::Contiguous])
 *   B1-->D
 * ```
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnMaskedScatterGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceMaskedScatter(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MASKED_SCATTER_H_
