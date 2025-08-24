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
#ifndef OP_API_INC_LEVEL2_ACLNN_RANDOM_H_
#define OP_API_INC_LEVEL2_ACLNN_RANDOM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInplaceRandom的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、INT32、INT64、BFLOAT16、FLOAT16、
 * INT16、INT8、UINT8、BOOL、DOUBLE。数据格式支持ND。支持非连续的Tensor。
 * @param [in] from: host侧的浮点型，进行离散均匀分布取值的左边界，输入为DOUBLE数据类型。
 * @param [in] to: host侧的浮点型，进行离散均匀分布取值的右边界，输入为DOUBLE数据类型。
 * @param [in] seed: 随机数生成器的种子,它影响生成的随机数序列，输入为UINT64_T数据类型。
 * @param [in] offset: 随机数生成器的偏移量,它影响生成的随机数序列的位置。设置偏移量后，
 * 生成的随机数序列会从指定位置开始。输入为UINT64_T数据类型。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceRandomGetWorkspaceSize(const aclTensor* selfRef, int64_t from, int64_t to,
                                                         int64_t seed, int64_t offset, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor);

/**
 * @brief aclnnInplaceRandom的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceRandom获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */

ACLNN_API aclnnStatus aclnnInplaceRandom(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_RANDOM_H_
