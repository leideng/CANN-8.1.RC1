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
#ifndef OP_API_INC_UNIQUE_CONSECUTIVE_H_
#define OP_API_INC_UNIQUE_CONSECUTIVE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUniqueConsecutive的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：从每个连续的相同元素组中消除除第一个元素之外的所有元素。
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 *  graph LR
 *      A[(self)] ---> B([l0op::Contiguous])
 *      B ---> F([l0op::UniqueConsecutive])
 *      C((returnInverse)) --->F
 *      D((returnCounts)) --->F
 *      E((dim)) --->F
 *      F --> G[(valueOut)]
 *      F --> H[(inverseOut)]
 *      F --> I[(countsOut)]
 * ```
 * @param [in] self：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、
 *                   UINT64、COMPLEX64、COMPLEX128、BOOL，数据格式支持ND。
 * @param [in] returnInverse: 表示是否返回self中各元素在valueOut中对应元素的位置下标，True时返回，False时不返回。
 * @param [in] returnCounts: 表示是否返回valueOut中各元素在self中连续重复出现的次数，True时返回，False时不返回。
 * @param [in] dim: 表示进行去重的维度。
 * @param [in] valueOut: 第一个输出张量，返回消除连续重复元素后的结果，数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、
 *                       INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX64、COMPLEX128、BOOL，数据格式支持ND。
 * @param [in] inverseOut:
 * 第二个输出张量，当returnInverse为True时有意义，返回self中各元素在valueOut中对应元素的位置下标，
 *                         数据类型支持INT64，数据格式支持ND。
 * @param [in] countsOut: 第三个输出张量，当returnCounts为True时有意义，返回valueOut中各元素在self中连续重复出现的次数，
 *                        数据类型支持INT64，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUniqueConsecutiveGetWorkspaceSize(const aclTensor* self, bool returnInverse,
                                                             bool returnCounts, int64_t dim, aclTensor* valueOut,
                                                             aclTensor* inverseOut, aclTensor* countsOut,
                                                             uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnUniqueConsecutive的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnUniqueConsecutiveGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUniqueConsecutive(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNIQUE_CONSECUTIVE_H_