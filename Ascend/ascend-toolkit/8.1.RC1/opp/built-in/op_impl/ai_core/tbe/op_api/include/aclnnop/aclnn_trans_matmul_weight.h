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
#ifndef OP_API_INC_TRANS_MATMUL_WEIGHT_H_
#define OP_API_INC_TRANS_MATMUL_WEIGHT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * aclnnCalculateMatmulWeightSizeV2用于计算调用aclnnMatMul、aclnnMm、aclnnWeightQuantBatchMatmulV2、
 * aclnnWeightQuantBatchMatmulV3、aclnnQuantMatmulV3传入的weight tensor需要占用的元素大小
 * @domain aclnn_ops_infer
 *
 * @param [in] tensorShape: 用于表达该次Matmul载入矩阵的weight的Shape
 * @param [in] dataType: 输入Weight的Datatype, 支持INT8和Float16
 * @param [out] weightTensorSize: 根据MatMul内部处理逻辑，计算该输入下Weight需要多少个元素的数据量
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCalculateMatmulWeightSizeV2(const aclIntArray* tensorShape, aclDataType dataType,
                                                       uint64_t* weightTensorSize);

/**
 * @brief aclnnCalculateMatmulWeightSize用于计算调用aclnnMatMul或aclnnMm传入的weight tensor需要占用的元素大小
 *
 *
 * @param [in] tensorShape: 用于表达该次Matmul载入矩阵的weight的Shape
 * @param [in] weightTensorSize: 根据MatMul内部处理逻辑，计算该输入下Weight需要多少个元素的数据量
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCalculateMatmulWeightSize(const aclIntArray* tensorShape, uint64_t* weightTensorSize);

/**
 * @brief aclnnTransMatmulWeight的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：将输入tensor转换为指定的dtype类型。
 *
 * @param [in] mmWeightRef: 输入是一个待处理的matmul的weightTensor，格式是正常的ND输入，数据类型支持Float16
 * 经过此接口处理后此tensor被刷新为预处理后的matmul weightTensor格式根据亲和性进行ND或者私有格式的转换
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnTransMatmulWeightGetWorkspaceSize(aclTensor* mmWeightRef, uint64_t* workspaceSize,
                                                             aclOpExecutor** executor);
/**
 * @brief aclnnTransMatmulWeight的第二段接口，用于执行计算。
 *
 * 算子功能：将输入tensor转换为指定的dtype类型。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnTransMatmulWeightGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnTransMatmulWeight(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TRANS_MATMUL_WEIGHT_H_
