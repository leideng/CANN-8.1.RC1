/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef OP_API_INC_ROPE_WITH_SIN_COS_CACHE_H_
#define OP_API_INC_ROPE_WITH_SIN_COS_CACHE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRopeWithSinCosCache的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 
 * 算子功能：完成rope计算。支持连续和非连续的输入。
 * 计算公式：
 * o1= x1 * cos - x2 * sin
 * o2= x2 * cos + x1 * sin
 * 其中x1、x2分别表示queryIn/keyIn的前半部分与后半部分，o1、o2分别表示queryOut/keyOut的前半部分与后半部分。
 * 参数描述：
 * @param [in] positions: device的aclTensor，为cossincache中的索引信息
 * @param [in] queryIn: device的aclTensor，表示要执行旋转位置编码的第一个张量，支持BFLOAT16，FLOAT16,FLOAT32
 * @param [in] keyIn: device的aclTensor，表示要执行旋转位置编码的第二个张量，支持BFLOAT16，FLOAT16,FLOAT32
 * @param [in] cosSinCache: device的aclTensor，表示参与计算的位置编码张量，支持BFLOAT16，FLOAT16,FLOAT32
 * @param [in] mropeSection: device的aclTensor，mrope模式下用于整合输入的位置编码张量信息;不使能mrope模式（即rope模式）输入为nullptr
 * @param [in] headSize: 属性，表示每个注意力头维度大小，数据类型int64
 * @param [in] isNeoxStyle: 属性，true表示rotate_half（GPT-NeoX style）计算模式，false表示rotate_interleaved（GPT-J style）计算模式，数据类型bool
 * @param [out] queryOut: device的aclTensor，输出query执行旋转位置编码后的结果, 类型同queryIn
 * @param [out] keyOut: device的aclTensor，输出key执行旋转位置编码后的结果，类型同keyIn
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRopeWithSinCosCacheGetWorkspaceSize(const aclTensor* positions, const aclTensor* queryIn,
                                                     const aclTensor* keyIn, const aclTensor* cosSinCache,
                                                     const aclIntArray* mropeSection, int64_t headSize,
                                                     bool isNeoxStyle, aclTensor* queryOut, aclTensor* keyOut,
                                                     uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnRopeWithSinCosCache的第二段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：完成rope计算。支持连续和非连续的输入
 * 参数描述：
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnRsubsGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRopeWithSinCosCache(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ROPE_WITH_SIN_COS_CACHE_NORM_H_
