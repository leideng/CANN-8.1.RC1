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
#ifndef OP_API_INC_ACLNN_FOREACH_ADDCDIV_SCALAR_V2_H_
#define OP_API_INC_ACLNN_FOREACH_ADDCDIV_SCALAR_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnForeachAddcdivScalarV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：对输入多个张量进行逐元素加、乘、除操作。
 * 计算公式：
 * out_i = {x}_{1i}+ \frac{{x}_{2i}}{{x}_{3i}}\times{scalar}
 * @domain aclnnop_math
 * 参数描述：
 * @param [in]   input
 * 输入Tensor，数据类型支持FLOAT、FLOAT16、BFLOAT16。数据格式支持ND。
  * @param [in]   input
 * 输入Tensor，数据类型支持FLOAT，FLOAT16，BFLOAT16。数据格式支持ND。
  * @param [in]   input
 * 输入Tensor，数据类型支持FLOAT，FLOAT16，BFLOAT16。数据格式支持ND。
  * @param [in]  input
 * 输入Scalar，数据类型支持FLOAT、FLOAT16。数据格式支持ND。
 * @param [in]   out
 * 输出Tensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。数据格式支持ND。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */
ACLNN_API aclnnStatus aclnnForeachAddcdivScalarV2GetWorkspaceSize(
    const aclTensorList *x1,
    const aclTensorList *x2,
    const aclTensorList *x3,
    const aclScalar *scalar,
    aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnForeachAddcdivScalarV2的第二段接口，用于执行计算。
 * 功能描述：对输入多个张量进行逐元素加、乘、除操作。
 * 计算公式：
 * out_i = {x}_{1i}+ \frac{{x}_{2i}}{{x}_{3i}}\times{scalar}
 * @domain aclnnop_math
 * 参数描述：
 * param [in] workspace: 在npu device侧申请的workspace内存起址。
 * param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnForeachAddcdivScalarV2GetWorkspaceSize获取。
 * param [in] stream: acl stream流。
 * param [in] executor: op执行器，包含了算子计算流程。
 * return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnForeachAddcdivScalarV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
