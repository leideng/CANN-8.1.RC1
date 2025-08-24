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

#ifndef OP_API_INC_LINALG_VECTOR_NORM_H_
#define OP_API_INC_LINALG_VECTOR_NORM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLinalgVectorNorm的第一段接口， 根据具体的计算流程，计算workspace大小
 * @domain aclnn_ops_infer
 * @param [in] self: npu device侧的aclTensor, 数据类型支持浮点数据类型
 * @param [in] ord: host侧aclScalar, 数据类型支持浮点数据类型
 * @param [in] dims: npu device侧的aclIntArray, 代表对应的降维轴Axis的信息
 * @param [in] keepDims: host侧的bool, 代表是否保留对应dim维度的信息
 * @param [in] dtype: 指定self计算时的数据类，在计算前将self转换成dtype指定类型进行计算，数据类型支持FLOAT、FLOAT16。
 * @param [in] out: 输出tensor
 * @param [in] workspaceSize: 返回用户需要的npu device侧申请的workspace大小
 * @param [in] executor: 返回op执行器，包含算子计算流程
 * @return aclnnStatus: 返回状态码
 * */
ACLNN_API aclnnStatus aclnnLinalgVectorNormGetWorkspaceSize(const aclTensor* self, const aclScalar* ord,
                                                            const aclIntArray* dims, bool keepDims,
                                                            const aclDataType dtype, aclTensor* out,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * aclnnLinalgVectorNorm的第二段接口，用于执行计算
 * @param [in] workspace: 在npu device侧申请的workspace内存地址
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，
 * 由第一段接口aclnnLinalgVectorNormGetWorkspaceSize获取
 * @param [in] executor: 返回op执行器，包含算子计算流程
 * @param [in] stream: acl stream流
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnLinalgVectorNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LINALG_VECTOR_NORM_H_