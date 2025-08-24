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
#ifndef OP_API_INC_LEVEL2_ACLNN_SPLIT_WITH_SIZE_H_
#define OP_API_INC_LEVEL2_ACLNN_SPLIT_WITH_SIZE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSplitWithSize的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE、BFLOAT16、INT32、INT64、INT16、INT8、UINT8、
 * BOOL、COMPLEX128和COMPLEX64。支持非连续的Tensor，数据格式支持ND。
 * @param [in] splitSize：host侧的aclIntArray,
 * 表示需要split的各块大小，数据类型支持INT64和INT32。所有块的大小总和需要等于self 在dim维度上的shape大小。
 * @param [in] dim: host侧的整型，数据类型为INT64，表示输入tensor被split的维度。
 * @param [in] out: npu device侧的aclTensorList，表示被split后的输出tensor的列表，数据类型支持FLOAT、FLOAT16、DOUBLE、
 * BFLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128和COMPLEX64。支持，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSplitWithSizeGetWorkspaceSize(const aclTensor* self, const aclIntArray* splitSize,
                                                         int64_t dim, aclTensorList* out, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor);

/**
 * @brief aclnnSplitWithSize的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSplitWithSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSplitWithSize(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_SPLIT_WITH_SIZE_H_
