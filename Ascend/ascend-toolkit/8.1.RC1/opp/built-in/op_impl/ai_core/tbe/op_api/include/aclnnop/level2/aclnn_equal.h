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
#ifndef OP_API_INC_TENSOREQUAL_H_
#define OP_API_INC_TENSOREQUAL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnEqual的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 计算两个Tensor是否有相同的大小和元素，返回一个Bool类型：
 *
 * $$ out = (self == other)  ?  True : False $$
 *
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([l0op::Contiguous])
 *     B -->D([l0op::TensorEqual])
 *     E[(other)] -->F([l0op::Contiguous])
 *     F -->D --> F1([l0op::ViewCopy])
 *     F1 --> J[(out)]
 * ```
 *
 * @param [in] self: npu device侧的aclTensor，
 * 数据类型支持FLOAT16,FLOAT,INT32,INT8,UINT8,BOOL,DOUBLE,INT64,INT16,UINT16,UINT32,UINT64数据类型，
 * self与other数据类型一致，支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: npu device侧的aclTensor，
 * 数据类型支持FLOAT16,FLOAT,INT32,INT8,UINT8,BOOL,DOUBLE,INT64,INT16,UINT16,UINT32,UINT64数据类型，
 * self与other数据类型一致，支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: 输出一个数据类型为BOOL类型的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEqualGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnEqual的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnEqualGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEqual(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TENSOREQUAL_H_
