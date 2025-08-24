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

/*!
 * \file cache_management_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_CACHE_MANAGEMENT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CACHE_MANAGEMENT_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 *@brief Operators for managing cache memory.

 *@par Inputs:
 *src: A ND Tensor with TensorType::NumberType().

 *@par Attributes:
 *@li max_size: The maximum memory size required for caching operation.
 *@li type: An optional int32 or int64 which has a default value of 6, indicating a prefetch operation.
 *@li offset: An optional int32 or int64 specifies the offset of the CMO operation address, which must not exceed the
 *size of the input memory. \n
 */
REG_OP(Cmo)
    .INPUT(src, TensorType::NumberType())
    .REQUIRED_ATTR(max_size, Int)
    .ATTR(type, Int, 6)
    .ATTR(offset, Int, 0)
    .OP_END_FACTORY_REG(Cmo)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_CACHE_MANAGEMENT_OPS_H_
