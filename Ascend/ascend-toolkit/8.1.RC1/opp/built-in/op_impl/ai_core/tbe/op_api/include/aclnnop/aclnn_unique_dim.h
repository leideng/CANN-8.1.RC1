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
#ifndef OP_API_INC_UNIQUE_DIM_H_
#define OP_API_INC_UNIQUE_DIM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUniqueDim的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：在某一dim轴上，对输入张量self做去重操作。。
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 *  graph LR
 *   A[(self)] ---> B([l0op::Contiguous])
 *   B ---> F([l0op::UniqueDim])
 *   J((sorted)) ---> F
 *   C((returnInverse)) --->F
 *   E((dim)) --->F
 *   F --> G[(valueOut)] --> F
 *   F --> H[(inverseOut)] -->F
 *   F --> I[(countsOut)] --> F
 * ```
 * @param [in]
 * self：Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、UINT8、INT8、UINT16、INT16、UINT32、INT32、UINT64、
 *                   INT64、FLOAT64 。支持非连续的Tensor，数据格式支持ND。
 * @param [in] sorted: 表示返回的输出结果valueOut是否排序。
 * @param [in] returnInverse: 表示是否返回self在dim轴上各元素在valueOut中对应元素的位置下标，True时返回，False时不返回。
 * @param [in] dim: Host侧的整型，指定做去重操作的维度，数据类型支持INT64，取值范围为[-self.dim(), self.dim())。
 * @param [in] valueOut:
 * 第一个输出张量，表示去重结果，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、UINT8、INT8、UINT16、
 *                       INT16、UINT32、INT32、UINT64、INT64、FLOAT64。支持非连续的Tensor，数据格式支持ND。
 * @param [in] inverseOut: 第二个输出张量，表示self在dim轴上各元素在valueOut中对应元素的位置下标，数据类型支持INT64。
 * @param [in] countsOut: 第三个输出张量，表示valueOut中的各元素在self中出现的次数，数据类型支持INT64。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUniqueDimGetWorkspaceSize(const aclTensor* self, bool sorted, bool returnInverse,
                                                     int64_t dim, aclTensor* valueOut, aclTensor* inverseOut,
                                                     aclTensor* countsOut, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor);

/**
 * @brief aclnnUniqueDim的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnUniqueDim获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUniqueDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNIQUE_DIM_H_