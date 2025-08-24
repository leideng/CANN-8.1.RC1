/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef OP_API_INC_LEVEL2_ACLNN_SILENT_CHECK_H_
#define OP_API_INC_LEVEL2_ACLNN_SILENT_CHECK_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSilentCheck的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * 算子功能： 根据输入val判断是否触发静默检测错误。
 * 该算子为自定义算子语义, 无对应的tensorflow或pytorch接口。
 * @param [in] val: npu device侧的aclTensor，数据类型支持FLOAT32, FLOAT16, BFLOAT16。shape为[1]。
 * @param [in] inputGrad: npu device侧的aclTensor，数据类型支持FLOAT32, FLOAT16, BFLOAT16。
 * @param [in] sfda: npu device侧的aclTensor，数据类型支持FLOAT32。shape为[3]。
 * @param [in] step: npu device侧的aclTensor，数据类型支持INT64。shape为[1]。
 * @param [in] cMinSteps: 需要累积的步数，数据类型为INT32。
 * @param [in] cThreshL1: 数值上的L1阈值，数据类型为FLOAT。
 * @param [in] cCoeffL1: 跳变上的L1阈值，数据类型为FLOAT。
 * @param [in] cThreshL2: 数值上的L2阈值，数据类型为FLOAT。
 * @param [in] cCoeffL2: 跳变上的L2阈值，数据类型为FLOAT。
 * @param [in] npuAsdDetect: 环境变量NPU_ASD_DETECT，数据类型为INT32。
 * @param [in] result: 判断是否触发静默检测故障结果的aclTensor，数据类型为INT32。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnSilentCheckGetWorkspaceSize(const aclTensor *val, aclTensor *inputGradRef,
                                                       aclTensor *sfdaRef, aclTensor *stepRef, const int32_t cMinSteps,
                                                       const float cThreshL1, const float cCoeffL1,
                                                       const float cThreshL2, const float cCoeffL2,
                                                       const int32_t npuAsdDetect, aclTensor* result,
                                                       uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief: aclnnSilentCheck的第二段接口，用于执行计算
 * @domain aclnn_ops_train
 * 算子功能： 根据输入val判断是否触发静默检测错误。
 * 该算子为自定义算子语义, 无对应的tensorflow或pytorch接口。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSilentCheckGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSilentCheck(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_SILENT_CHECK_H_