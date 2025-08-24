/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/license/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_BINARY_CROSS_ENTROPY_WITH_LOGITS_BACKEARD_H_
#define OP_API_INC_BINARY_CROSS_ENTROPY_WITH_LOGITS_BACKEARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize的第一段接口，根据具体的计算流程，计算workspace大小
 * @domain aclnn_ops_train
 * 算子功能：求二元交叉熵反向传播的梯度值
 */
ACLNN_API aclnnStatus aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* weightOptional,
    const aclTensor* posWeightOptional, int64_t reduction, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/*
 * @brief aclnnBinaryCrossEntropyWithLogitsBackward的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnBinaryCrossEntropyWithLogitsBackward(void* workspace, uint64_t workspaceSize,
                                                                aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BINARY_CROSS_ENTROPY_WITH_LOGITS_BACKEARD_H_
