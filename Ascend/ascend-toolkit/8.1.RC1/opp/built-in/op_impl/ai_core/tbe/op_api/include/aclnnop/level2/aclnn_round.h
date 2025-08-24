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
#ifndef OP_API_INC_ROUND_H_
#define OP_API_INC_ROUND_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRound的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：将输入的值舍入到最接近的整数，若该值与两个整数距离一样则向偶数取整。
 * @param [in] self: npu device侧的aclTensor，数据类型支持BFLOAT16,FLOAT16, FLOAT32, DOUBLE, INT32, INT64，
 * 数据格式支持ND，shape不支持9D及以上。
 * @param [in] out: npu device侧的aclTensor，数据类型、shape与self一致，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRoundGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * @brief: aclnnRound的第二段接口，用于执行计算
 *
 * 算子功能：将输入的值舍入到最接近的整数，若该值与两个整数距离一样则向偶数取整。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnRoundGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRound(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                 const aclrtStream stream);

/**
 * @brief aclnnInplaceRound的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：将输入的值舍入到最接近的整数，若该值与两个整数距离一样则向偶数取整。
 * @param [in] selfRef: npu device侧的aclTensor，数据类型支持FLOAT16, FLOAT32, DOUBLE, INT32, INT64，
 * 数据格式支持ND，shape不支持9D及以上。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnInplaceRoundGetWorkspaceSize(const aclTensor* selfRef, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor);

/**
 * @brief: aclnnInplaceRound的第二段接口，用于执行计算
 *
 * 算子功能：将输入的值舍入到最接近的整数，若该值与两个整数距离一样则向偶数取整。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSinGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceRound(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        const aclrtStream stream);

/**
 * @brief aclnnRoundDecimals的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 功能描述：将输入Tensor的元素四舍五入到指定的位数。
 * 参数描述：
 * @param [in]   self
 * 输入Tensor，数据类型支持FLOAT，BFLOAT16,FLOAT16，DOUBLE，INT32，INT64。支持非连续Tensor，数据格式支持ND。
 * @param [in]   decimals: 需要四舍五入到的位数，数据类型支持INT。
 * @param [in]   out
 * 输出Tensor，数据类型支持FLOAT，FLOAT16，DOUBLE，INT32，INT64。支持非连续Tensor，数据格式支持ND。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */
ACLNN_API aclnnStatus aclnnRoundDecimalsGetWorkspaceSize(const aclTensor* self, int64_t decimals, aclTensor* out,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnRoundDecimals的第二段接口，用于执行计算。
 * 功能描述：将输入Tensor的元素四舍五入到指定的位数。
 * 实现说明：
 * api计算的基本路径：
```mermaid
graph LR
    A[(Self)] -->B([l0op::Contiguous])
    B --> D([l0op::Round])

    C((decimals)) --> D([l0op::Cast])
    D --> E([l0op::ViewCopy])
    E --> F[(Out)]
```
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnRoundDecimalsGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRoundDecimals(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

/**
 * @brief aclnnInplaceRoundDecimals的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 功能描述：将输入Tensor的元素四舍五入到指定的位数。
 * 参数描述：
 * @param [in]   selfRef
 * 输入Tensor，数据类型支持FLOAT，FLOAT16，DOUBLE，INT32，INT64。支持非连续Tensor，数据格式支持ND。
 * @param [in]   decimals: 需要四舍五入到的位数，数据类型支持INT。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */
ACLNN_API aclnnStatus aclnnInplaceRoundDecimalsGetWorkspaceSize(aclTensor* selfRef, int64_t decimals,
                                                                uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnInplaceRoundDecimals的第二段接口，用于执行计算
 * 算子功能： 将输入Tensor的元素四舍五入到指定的位数。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceRoundDecimalsGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceRoundDecimals(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ROUND_H_