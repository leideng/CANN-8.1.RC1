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
#ifndef OP_API_INC_FRAC_H_
#define OP_API_INC_FRAC_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFrac的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 功能描述：计算输入Tensor中每个元素的小数部分。
 * 计算公式：
 * out_{i}=input_{i} - \lfloor \vert input_{i} \vert \rfloor * sgn(input_{i})
 * 参数描述：
 * @param [in]   input
 * 输入Tensor，数据类型支持FLOAT16、FLOAT、UINT8、INT8、INT16、INT32、INT64。支持非连续Tensor，数据格式支持ND。
 * @param [in]   out
 * 输出Tensor，数据类型支持FLOAT16、FLOAT、UINT8、INT8、INT16、INT32、INT64。支持非连续Tensor，数据格式支持ND。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */
ACLNN_API aclnnStatus aclnnFracGetWorkspaceSize(const aclTensor* input, aclTensor* out, uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief aclnnFrac的第二段接口，用于执行计算。
 * 功能描述：计算输入Tensor中每个元素的小数部分。
 * 计算公式：
 * out_{i}=input_{i} - \lfloor \vert input_{i} \vert \rfloor * sgn(input_{i})
 * 实现说明：
 * api计算的基本路径：
```mermaid
graph LR
    A[(Input)] -->B([l0op::Contiguous])
    B -->C([l0op::Sub])

    B -->F1([l0op::Trunc])
    F1 --> C
    C -->D([l0op::ViewCopy])
    D -->E[(Out)]
```
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnFracGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFrac(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceFrac的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 功能描述：计算输入Tensor中每个元素的小数部分。
 * 参数描述：
 * @param [in]   inputRef
 * 输入Tensor，数据类型支持FLOAT16、FLOAT、UINT8、INT8、INT16、INT32、INT64。支持非连续Tensor，数据格式支持ND。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */
ACLNN_API aclnnStatus aclnnInplaceFracGetWorkspaceSize(aclTensor* inputRef, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief: aclnnInplaceFrac的第二段接口，用于执行计算
 * 功能描述：计算输入Tensor中每个元素的小数部分。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceFracGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceFrac(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
