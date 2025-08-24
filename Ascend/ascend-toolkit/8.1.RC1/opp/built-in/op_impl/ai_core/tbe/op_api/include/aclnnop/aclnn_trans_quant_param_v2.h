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
#ifndef OP_API_INC_TRANS_QUANT_PARAM_V2_H
#define OP_API_INC_TRANS_QUANT_PARAM_V2_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现transQuantParamV2计算
 * @brief aclnnTransQuantParamV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] scale: 量化参数，数据类型支持： float32。
 * @param [in] offset: 量化参数，数据类型支持：float32。
 * @param [out] out: 计算结果，数据类型：uint64_t
 * @return aclnnStatus: 返回状态码
 */

ACLNN_API aclnnStatus aclnnTransQuantParamV2GetWorkspaceSize(const aclTensor* scale, const aclTensor* offset,
                                                             const aclTensor* out, uint64_t* workspaceSize,
                                                             aclOpExecutor** executor);

/**
 * @brief aclnnTransQuantParamV2的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnTransQuantParamV2GetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnTransQuantParamV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TRANS_QUANT_PARAM_V2_H