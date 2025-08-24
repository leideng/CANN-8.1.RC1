/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OP_API_INC_NON_MAX_SUPPRESSION_H_
#define OP_API_INC_NON_MAX_SUPPRESSION_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnNonMaxSuppressionGetWorkspaceSize(const aclTensor* boxes, const aclTensor* scores,
                                                             aclIntArray* maxOutputBoxesPerClass,
                                                             aclFloatArray* iouThreshold, aclFloatArray* scoreThreshold,
                                                             int32_t centerPointBox, aclTensor* selectedIndices,
                                                             uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnNonMaxSuppression(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_NON_MAX_SUPPRESSION_H_