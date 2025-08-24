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
#ifndef OP_API_INC_LEVEL2_ACLNN_ERFC_H_
#define OP_API_INC_LEVEL2_ACLNN_ERFC_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnErfc的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：返回输入Tensor中每个元素对应的误差互补函数的值。
 * 计算公式：
 * $$ erfc(x)=1-\frac{2}{\sqrt{\pi } } \int_{0}^{x} e^{-t^{2} } \mathrm{d}t $$
 *
 * 场景：当输入类型在Erfc算子支持的范围之内（FLOAT32、BFLOAT16、FLOAT16、FLOAT64）时，使用Erfc算子完成计算。
 * 计算图：
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B([l0op::Contiguous])
 *     B --> C([l0op::Erfc])
 *     C --> D([l0op::Cast])
 *     D --> E([l0op::ViewCopy])
 *     E --> F[(out)]
 * ```
 *
 * 场景：self的数据类型为BOOL或INT64，将self的数据类型CAST为FLOAT32，再使用Erfc算子完成计算。
 * 计算图：
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B([l0op::Contiguous])
 *     B -->C([l0op::Cast])
 *     C -->D([l0op::Erfc])
 *     D --> E([l0op::Cast])
 *     E --> F([l0op::ViewCopy])
 *     F --> G[(out)]
 * ```
 *
 * @param [in] self: 待进行erfc计算的入参。npu device侧的aclTensor，
 * 数据类型支持FLOAT64、FLOAT32、BFLOAT16、FLOAT16、BOOL、INT64，数据格式支持ND， 支持非连续的Tensor。
 * @param [in] out: erfc计算的出参。npu device侧的aclTensor，
 * 数据类型支持FLOAT64、FLOAT32、BFLOAT16、FLOAT16，默认和self保持一致，若self数据类型为BOOL或INT64时，out的数据类型默认为FLOAT32，
 * 数据格式支持ND，和self的shape保持一致，支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnErfcGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief aclnnErfc的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnErfcGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnErfc(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                const aclrtStream stream);

/**
 * @brief aclnnInplaceErfc的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：返回输入Tensor中每个元素对应的误差互补函数的值
 * 计算公式：
 * $$ erfc(x)=1-\frac{2}{\sqrt{\pi } } \int_{0}^{x} e^{-t^{2} } \mathrm{d}t $$
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B{l0op::Contiguous}
 *     B -->C([l0op::Erfc])
 *     C --> D{l0op::ViewCopy}
 *     D --> E[(out)]
 * ```
 *
 * @param [in] selfRef: 待进行erfc计算的入参。npu device侧的aclTensor，
 * 数据类型支持FLOAT64、FLOAT32、BFLOAT16、FLOAT16，数据格式支持ND， 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceErfcGetWorkspaceSize(const aclTensor* selfRef, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnInplaceErfc的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceErfcGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceErfc(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ERFC_H_