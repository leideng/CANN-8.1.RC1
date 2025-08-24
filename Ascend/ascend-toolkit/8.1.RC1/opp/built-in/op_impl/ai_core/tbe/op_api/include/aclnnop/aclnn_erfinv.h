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
#ifndef OP_API_INC_LEVEL2_ACLNN_ERFINV_H_
#define OP_API_INC_LEVEL2_ACLNN_ERFINV_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnErfinv的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：erfinv是高斯误差函数erf的反函数。返回输入Tensor中每个元素对应在标准正态分布函数的分位数。
 * 计算公式：如下
 * $$
 * y = erfinv(x) \\
 * x = erf(y)=\frac{2}{\sqrt{\pi } } \int_{0}^{y} e^{-t^{2} } \mathrm{d}t
 * $$
 *
 * 计算图：如下
 * 场景：当输入类型在Erfinv算子支持的范围之内（FLOAT32、FLOAT16、BFLOAT16）时，使用Erfinv算子完成计算。
 * ```mermaid
 * graph LR
 * A[(Self)] --> B([l0op::Contiguous]) --> C([l0op::Erfinv])
 * C --> D([l0op::Cast]) --E D([l0op::ViewCopy]) --> F[(out)]
 * ```
 *
 * 整数类型：BOOL、INT8、INT16、INT32、INT64、UINT8，先转为FLOAT，再计算：
 * ```mermaid
 * graph LR
 * A[(Self)] --> B([l0op::Contiguous]) --> C([l0op::Cast]) --> D([l0op::Erfinv])
 * D --> E([l0op::Cast]) --> F([l0op::ViewCopy]) --> G[(out)]
 * ```
 *
 * @param [in] self: 待进行erfinv计算的入参。npu device侧的aclTensor，
 * 数据类型支持FLOAT32、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL，数据格式支持ND， 支持非连续的Tensor。
 * @param [in] out: erfinv计算的出参。npu device侧的aclTensor，
 * 数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND，shape同self一致，支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnErfinvGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);

/**
 * @brief aclnnErfinv的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnErfinvGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnErfinv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  const aclrtStream stream);

/**
 * @brief aclnnInplaceErfinv的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：返回输入Tensor中每个元素对应在标准正态分布函数的分位数
 *
 * @param [in] selfRef: 待进行erfinv计算的入参。npu device侧的aclTensor，
 * 数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND， 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceErfinvGetWorkspaceSize(const aclTensor* selfRef, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor);

/**
 * @brief aclnnInplaceinv的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnErfinvGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceErfinv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ERFINV_H_