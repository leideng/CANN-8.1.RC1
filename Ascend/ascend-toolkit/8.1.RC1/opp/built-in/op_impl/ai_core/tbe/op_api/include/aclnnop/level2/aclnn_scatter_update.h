/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_SCATTER_UPDATE_H_
#define OP_API_INC_SCATTER_UPDATE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnScatterUpdate的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 将tensor updates中的值按指定的轴axis和索引indices逐个更新tensor data中的值。该算子为自定义算子语义,
 * 无对应的tensorflow或pytorch接口。
 * @param [in] data: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, INT8, BFLOAT16,data只支持四维，
 * 且维度大小和维度数量需要与updates一致。支持非连续的Tensor，数据格式支持ND。
 * @param [in] indices: npu device侧的aclTensor，数据类型支持INT32,
 * int64类型，目前仅支持一维。支持非连续的Tensor，数据格式支持ND。
 * @param [in] updates: npu device侧的aclTensor，数据类型支持FLOAT16, FLOAT32, INT8,
 * BFLOAT16,且维度大小和维度数量需要与data一致。 支持非连续的Tensor，数据格式支持ND,
 * @param [in] axis: host侧的axis, 数据类型支持INT64。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnInplaceScatterUpdateGetWorkspaceSize(aclTensor* data, const aclTensor* indices,
                                                                const aclTensor* updates, int64_t axis,
                                                                uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnScatterUpdate的第二段接口，用于执行计算
 *
 * 算子功能: 将tensor updates中的值按指定的轴axis和索引indices逐个更新tensor data中的值。该算子为自定义算子语义,
 * 无对应的tensorflow或pytorch接口。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceScatterUpdateGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceScatterUpdate(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SCATTER_UPDATE_H_