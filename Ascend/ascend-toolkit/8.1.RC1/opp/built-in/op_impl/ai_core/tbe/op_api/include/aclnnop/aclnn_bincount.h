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
#ifndef OP_API_INC_BINCOUNT_H_
#define OP_API_INC_BINCOUNT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBincount 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：计算非负整数数组中每个数的频率。
 * 计算公式：
 * 如果n是self在位置i上的值，如果指定了weights，则
 * out[n] = out[n] + weights[i]
 * 否则
 * out[n] = out[n] + 1
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持INT8、INT16、INT32、INT64、UINT8，
 * 且必须是非负整数，数据格式支持1维ND。支持非连续的Tensor。
 * @param [in] weights:  npu device侧的aclTensor，self每个值的权重，可为空指针。
 * 数据类型支持FLOAT、FLOAT16、FLOAT64、INT8、INT16、INT32、INT64、UINT8、BOOL，
 * 数据格式支持1维ND，且shape必须与self一致。支持非连续的Tensor。
 * @param [in] minlength: host侧的int型，指定输出tensor最小长度。如果计算出来的size的最大值小于minlength，
 * 则输出长度为minlength，否则为size。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持INT32、INT64、FLOAT、DOUBLE。数据格式支持1维ND。支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBincountGetWorkspaceSize(const aclTensor* self, const aclTensor* weights, int64_t minlength,
                                                    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnBincount的第一段接口，根据具体的计算流程，计算workspace大小。
 *
 * 算子功能：计算非负整数数组中每个数的频率。
 * 计算公式：
 * 如果n是self在位置i上的值，如果指定了weights，则
 * out[n] = out[n] + weights[i]
 * 否则
 * out[n] = out[n] + 1

 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnBincountGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBincount(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BINCOUNT_H_