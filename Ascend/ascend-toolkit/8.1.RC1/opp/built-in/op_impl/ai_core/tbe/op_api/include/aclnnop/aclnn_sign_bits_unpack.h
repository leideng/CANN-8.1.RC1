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
#ifndef OP_API_INC_SIGN_BITS_UNPACK_H_
#define OP_API_INC_SIGN_BITS_UNPACK_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSignBitsUnpack的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnnop_ops_train
 *
 * 算子功能：将uint8类型1位Adam拆包为float32或者float16。
 *
 * @param [in] self: 
 * device侧的aclTensor，数据类型支持UINT8，支持空Tensor。支持非连续的Tensor，数据类型支持UINT8，数据格式支持ND。
 * @param [in] size:    host侧的int64_t，reshape时输出张量的第一个维度。
 * @param [in] dtype: 	host侧的aclDataType，表示量化输出Tensor的数据类型，数据类型支持ACL_FLOAT16、ACL_FLOAT。
 * @param [in] out: 
 * device侧的aclTensor，数据类型支持FLOAT16、FLOAT。数据类型由dtype决定，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSignBitsUnpackGetWorkspaceSize(const aclTensor* self, int64_t size, aclDataType dtype, aclTensor* out,
                                                uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnSignBitsUnpack的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSignBitsUnpackGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSignBitsUnpack(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SIGN_BITS_UNPACK_H_
