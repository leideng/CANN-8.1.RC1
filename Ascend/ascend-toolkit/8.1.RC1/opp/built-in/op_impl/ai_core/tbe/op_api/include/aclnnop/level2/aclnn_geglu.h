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
#ifndef OP_API_INC_LEVEL2_ACLNN_GEGLU_H_
#define OP_API_INC_LEVEL2_ACLNN_GEGLU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGeGlu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：高斯误差线性单元激活函数
 * 计算公式：
 * $$ out_{i}=GeGlu(self_{i}) = A \cdot Gelu(B) $$
 * 其中，$A$表示$self$的前半部分，$B$表示$self$的后半部分。
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Contiguous])
 *     B --> C([l0op::GeGlu])
 *     C --> D([l0op::ViewCopy])
 *     D --> E[(out)]
 *     D --> F[(outGelu)]
 *
 *     G((dim)) --> C
 *     H((approximate)) --> C
```
 */

/**
 * @param [in] self: 待进行GeGlu计算的入参，npu
 * device侧的aclTensor，数据类型支持FLOAT32、FLOAT16、BFLOAT16（910B支持），支持
 * [非连续的Tensor](#)，数据格式支持ND([参考](#))。
 * @param [in] dim：可选入参，设定的slice轴，数据类型支持INT64。
 * @param [in] approximate：可选入参，GeGlu计算使用的激活函数索引，0表示使用`tanh`，1表示使用`none`，数据类型支持INT64。
 * @param [in] out: GeGlu计算的出参，npu
 * device侧的aclTensor，数据类型需要与self一致，支持[非连续的Tensor](#)，数据格式支持ND([参考](#))。
 * @param [in] outGelu: Gelu(B)计算的出参，npu
 * device侧的aclTensor，数据类型需要与self一致，支持[非连续的Tensor](#)，数据格式支持ND([参考](#))。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGeGluGetWorkspaceSize(const aclTensor* self, int64_t dim, int64_t approximate,
                                                 aclTensor* out, aclTensor* outGelu, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * @param [in] self: 待进行GeGlu计算的入参，npu
 * @domain aclnn_ops_infer
 * device侧的aclTensor，数据类型支持FLOAT32、FLOAT16、BFLOAT16（910B支持），支持
 * [非连续的Tensor](#)，数据格式支持ND([参考](#))。
 * @param [in] dim：可选入参，设定的slice轴，数据类型支持INT64。
 * @param [in] approximate：可选入参，GeGlu计算使用的激活函数索引，0表示使用`tanh`，1表示使用`none`，数据类型支持INT64。
 * @param [in] out: GeGlu计算的出参，npu
 * device侧的aclTensor，数据类型需要与self一致，支持[非连续的Tensor](#)，数据格式支持ND([参考](#))。
 * @param [in] activateLeft:
 * 计算属性，host侧的布尔值，表示激活函数操作数据块的方向，默认值为false，表示对右边做activate。
 * @param [in] outGelu: Gelu(B)计算的出参，npu
 * device侧的aclTensor，数据类型需要与self一致，支持[非连续的Tensor](#)，数据格式支持ND([参考](#))。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGeGluV3GetWorkspaceSize(const aclTensor* self, int64_t dim, int64_t approximate,
                                                   bool activateLeft, aclTensor* out, aclTensor* outGelu,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGeGlu的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGeGluGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGeGlu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnGeGlu的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGeGluGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGeGluV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GEGLU_H_
