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
#ifndef OP_API_INC_PUT_H_
#define OP_API_INC_PUT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInplacePut的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：将selfRef视为一维张量，把index张量中元素值作为索引，如果accumulate为true，把source中元素和selfRef对应的位置上元素做累加操作;
 * 如果accumulate为false，把source中元素替换掉selfRef对应位置上的元素。
 *
 * @param [in] selfRef: npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL，
 * 支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [in] index: npu
 * device侧的aclTensor，数据类型支持INT32、INT64,元素个数要求和source保持一致。支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与self一致。
 * @param [in] source: npu
 * device侧的aclTensor，数据类型和selfRef一致,元素个数为和index一致。支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplacePutGetWorkspaceSize(aclTensor* selfRef, const aclTensor* index,
                                                      const aclTensor* source, bool accumulate, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief aclnnInplacePut的第二段接口，用于执行计算。
 *
 * 算子功能：将selfRef视为一维张量，把index张量中元素值作为索引，如果accumulate为true，把source中元素和selfRef对应的位置上元素做累加操作;
 * 如果accumulate为false，把source中元素替换掉selfRef对应位置上的元素。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplacePutGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplacePut(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_PUT_H_
