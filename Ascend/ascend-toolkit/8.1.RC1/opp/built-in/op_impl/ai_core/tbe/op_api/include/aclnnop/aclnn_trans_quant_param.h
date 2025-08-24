
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
#ifndef OP_API_INC_LEVEL2_ACLNN_TRANS_QUANT_PARAM_H_
#define OP_API_INC_LEVEL2_ACLNN_TRANS_QUANT_PARAM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 将scale数据从float类型转换为硬件需要的uint64_t类型
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnTransQuantParam(const float* scaleArray, uint64_t scaleSize, const float* offsetArray,
                                           uint64_t offsetSize, uint64_t** quantParam, uint64_t* quantParamSize);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_TRANS_QUANT_PARAM_H_
