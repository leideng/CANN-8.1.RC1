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
#ifndef OP_API_INC_SORT_H_
#define OP_API_INC_SORT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSort的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 对输入Tensor完成sort操作
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8。支持空Tensor，
 * 支持[非连续的Tensor](#)。
 * @param [in] stable: 是否稳定排序, True为稳定排序，False为非稳定排序, 数据类型为BOOLEAN。
 * @param [in] dim: 用来作为排序标准的维度, 数据类型为INT。范围为 [-N, N-1]。
 * @param [in] descending: 控制排序顺序，True为降序，False为升序, 数据类型为BOOLEAN。
 * @param [in] valuesOut: npu device侧的aclTensor, 数据类型支持FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8。支持
 * 空Tensor，支持[非连续的Tensor](#)。数据格式支持ND([参考](#))。shape和数据格式需要与self一致。
 * @param [in] indicesOut: npu device侧的aclTensor,
 * 数据类型支持INT64。支持空Tensor，支持[非连续的Tensor](#)。数据格式支持 ND([参考](#))。shape和数据格式需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnSortGetWorkspaceSize(const aclTensor* self, bool stable, int64_t dim, bool descending,
                                                aclTensor* valuesOut, aclTensor* indicesOut, uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief: aclnnSort的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成sort操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSortGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSort(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
