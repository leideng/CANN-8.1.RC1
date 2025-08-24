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

#ifndef OP_API_INC_LEVEL0_RESHAPE_H
#define OP_API_INC_LEVEL0_RESHAPE_H

#include "opdev/shape_utils.h"
#include "opdev/op_def.h"

namespace l0op {
/**
 * @brief Modify input tensor's shape.
 * @param x Input Tensor. Should be contiguous.
 * @param shape Target Shape. Only one dimension can be -1.
 * @param executor aclOpExecutor.ldd
 * @return *aclTensor Output tensor.
 */
const aclTensor *Reshape(const aclTensor *x, const op::Shape& shape, aclOpExecutor *executor);

/**
 * @brief Modify input tensor's shape.
 * @param x Input Tensor. Should be contiguous.
 * @param shape Target Shape. Only one dimension can be -1.
 * @param executor aclOpExecutor.
 * @return *aclTensor Output tensor.
 */
const aclTensor *Reshape(const aclTensor *x, const aclIntArray* shape, aclOpExecutor *executor);
} // l0op

#endif  // OP_API_INC_LEVEL0_RESHAPE_H
