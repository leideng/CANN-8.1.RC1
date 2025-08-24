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
#ifndef OP_API_INC_SCATTER_OUT_H_
#define OP_API_INC_SCATTER_OUT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnScatterAdd的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 将源tensor中的值按指定的轴方向和index tensor中的位置关系逐个填入输出tensor中，
 * 若有多于一个src值被填入到self的同一位置，那么这些值将会在这一位置上进行累加
 * @param [in] self: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, INT32, INT8, UINT8,
 * 支持非连续的Tensor，数据格式支持ND,
 * @param [in] dim: host侧的num, 数据类型支持INT64。
 * @param [in] index: npu device侧的aclTensor，数据类型支持INT32, int64类型，dim反向的维度数量需要与src相同。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] src: npu device侧的aclTensor，数据类型支持FLOAT16, FLOAT32, INT32, INT8,
 * UINT8类型，dim反向的维度数量需要与src相同。 支持非连续的Tensor，数据格式支持ND，且数据类型与self保持一致。
 * @param [in] out: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, INT32, INT8, UINT8,
 * 数据类型,数据格式,tensor shape需要与self保持一致
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnScatterAddGetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* index,
                                                      const aclTensor* src, aclTensor* out, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief: aclnnScatterAdd的第二段接口，用于执行计算
 *
 * 算子功能: 将源tensor中的值按指定的轴方向和index tensor中的位置关系逐个填入输出tensor中，
 * 若有多于一个src值被填入到self的同一位置，那么这些值将会在这一位置上进行累加
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnScatterAddGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnScatterAdd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SCATTER_OUT_H_