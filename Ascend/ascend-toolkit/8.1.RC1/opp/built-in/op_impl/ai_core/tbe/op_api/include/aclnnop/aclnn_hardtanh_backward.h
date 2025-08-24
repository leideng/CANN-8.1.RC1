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
#ifndef OP_API_INC_HARDTANH_BACKWARD_H_
#define OP_API_INC_HARDTANH_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnHardtanhBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：完成Hardtanh的反向计算
 *
 * 实现说明：api
 * 计算的基本路径：如下所示
 * ```mermaid
 * graph LR
 *     A[(gradOutput)] -->B([l0op::Contiguous])
 *     B -->C([l0op::HardtanhGrad])
 *     D[(self)] -->E([l0op::Contiguous])
 *     E -->C([l0op::HardtanhGrad])
 *     F((min)) --> C([l0op::HardtanhGrad])
 *     G((max)) --> C([l0op::HardtanhGrad])
 *     C --> H([l0op::ViewCopy])
 *     H --> K[(out)]
 * ```
 *
 * @param [in] gradOutput: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16(仅昇腾910B和910_93
 * AI处理器支持)，支持非连续的Tensor，数据格式支持ND。
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16(仅昇腾910B和910_93
 * AI处理器支持)，支持非连续的Tensor，数据格式支持ND。
 * @param [in] min: 下界。
 * @param [in] max: 上界。
 * @param [out] out: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16(仅昇腾910B和910_93
 * AI处理器支持)，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnHardtanhBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                            const aclScalar* min, const aclScalar* max, aclTensor* out,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnHardtanhBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 *
 * 算子功能：完成Hardtanh的反向计算
 *
 * 实现说明：api
 * 计算的基本路径：如下所示
 * ```mermaid
 * graph LR
 *     A[(gradOutput)] -->B([l0op::Contiguous])
 *     B -->C([l0op::HardtanhGrad])
 *     D[(self)] -->E([l0op::Contiguous])
 *     E -->C([l0op::HardtanhGrad])
 *     F((min)) --> C([l0op::HardtanhGrad])
 *     G((max)) --> C([l0op::HardtanhGrad])
 *     C --> H([l0op::ViewCopy])
 *     H --> K[(out)]
 * ```
 *
 * @param [in] gradOutput: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16(仅昇腾910B和910_93
 * AI处理器支持)，支持非连续的Tensor，数据格式支持ND。
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16(仅昇腾910B和910_93
 * AI处理器支持)，支持非连续的Tensor，数据格式支持ND。
 * @param [in] min: 下界。
 * @param [in] max: 上界。
 * @param [out] out: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16(仅昇腾910B和910_93
 * AI处理器支持)，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnHardtanhBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_HARDTANH_BACKWARD_H_
