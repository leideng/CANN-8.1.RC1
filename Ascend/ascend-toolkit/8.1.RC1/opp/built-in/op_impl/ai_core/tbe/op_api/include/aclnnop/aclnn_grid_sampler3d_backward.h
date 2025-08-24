/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023 All rights reserved.
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

#ifndef OP_API_INC_GRID_SAMPLER3D_BACKWARD_H_
#define OP_API_INC_GRID_SAMPLER3D_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGridSampler3DBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
ACLNN_API aclnnStatus aclnnGridSampler3DBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* input,
                                                                 const aclTensor* grid, int64_t interpolationMode,
                                                                 int64_t paddingMode, bool alignCorners,
                                                                 const aclBoolArray* outputMask, aclTensor* inputGrad,
                                                                 aclTensor* gridGrad, uint64_t* workspaceSize,
                                                                 aclOpExecutor** executor);

/**
 * @brief aclnnGridSampler3DBackward的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnGridSampler3DBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GRID_SAMPLER3D_BACKWARD_H_
