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
#ifndef OP_API_INC_MAX_V2_H_
#define OP_API_INC_MAX_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMaxV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：按指定维度对输入tensor求元素最大值。
 *
 * @param [in] self: device侧的aclTensor，数据类型支持整型、浮点类型。支持非连续的Tensor，数据格式支持ND。
 * @param [in] dims: host侧aclIntArray，指定了要进行最大值计算的维度， 数据类型支持INT32和INT64。
 * @param [in] keepDims: host侧的布尔型，是否在输出张量中保留输入张量的维度。
 * @param [in] noopWithEmptyDims: host侧的布尔型，定义dims为空时的行为。
 * @param [in] out: device侧的aclTensor，数据类型支持整型、浮点类型。支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMaxV2GetWorkspaceSize(const aclTensor* self, const aclIntArray* dims, bool keepDims,
                                                 bool noopWithEmptyDims, aclTensor* out, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * @brief aclnnMaxV2的第二段接口，用于执行计算。
 *
 * 算子功能：按指定维度对输入tensor求元素最大值。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnMaxV2GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMaxV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MAX_V2_H_
