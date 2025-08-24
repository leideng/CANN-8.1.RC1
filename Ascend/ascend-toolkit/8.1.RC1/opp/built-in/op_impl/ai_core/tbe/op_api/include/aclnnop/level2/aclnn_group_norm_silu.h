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
#ifndef OP_API_INC_GROUP_NORM_SILU_H_
#define OP_API_INC_GROUP_NORM_SILU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupNormSilu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：完成GroupNorm和Silu融合算子功能。
 * 计算公式是：
 * y=((x-Ex) / (sqrt(Var(x)+ϵ​x))​)∗ γ + β
 * 将channel方向分group，然后每个group内做归一化，算(C//G)HW的均值
 *
 * 附：Silu计算公式为：
 * f(x) = x * sigmoid(x)
 * 当x大于0时,Silu激活函数将会放大x,而当x小于0时,Silu激活函数将会降低x,可以抑制过拟合
 * ```
 *
 * @param [in] self：计算输入，shape大于等于2维
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND，支持非连续的Tensor。
 * @param [in] gamma：计算输入，shape为1维，且等于c维度。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND，支持非连续的Tensor。
 * @param [in] beta：shape为1维，且等于c维度。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND，支持非连续的Tensor。
 * @param [in] group：计算属性，host侧的整数，数据类型支持INT64,分组信息
 * @param [in] esp：计算属性，host侧的浮点数，数据类型支持double,默认le-5
 * @param [out] out：y的输出。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND。
 * @param [out] meanOut：均值的输出。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND。
 * @param [out] rstdOut：1/sqrt(Var(x)+ϵ​x)结果输出
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupNormSiluGetWorkspaceSize(const aclTensor* self, const aclTensor* gamma,
                                                         const aclTensor* beta, int64_t group, double eps,
                                                         aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGroupNormSilu的第二段接口，用于执行计算
 */
ACLNN_API aclnnStatus aclnnGroupNormSilu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         const aclrtStream stream);

/**
 * @brief aclnnGroupNormSiluV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnGroupNormSiluV2GetWorkspaceSize(const aclTensor* self, const aclTensor* gamma,
                                                           const aclTensor* beta, int64_t group, double eps,
                                                           bool activateSilu, aclTensor* out, aclTensor* meanOut,
                                                           aclTensor* rstdOut, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnGroupNormSiluV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GROUP_NORM_SILU_H_
