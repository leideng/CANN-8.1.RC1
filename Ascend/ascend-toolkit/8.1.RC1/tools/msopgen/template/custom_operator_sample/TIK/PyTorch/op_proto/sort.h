/*
 * Copyright (C)  2020. Huawei Technologies Co., Ltd. All rights reserved.
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
 * \file sort.h
 * \brief
 */
#ifndef GE_OP_Sort_H
#define GE_OP_Sort_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
REG_OP(Sort)
    .INPUT(x, TensorType({ DT_FLOAT16 }))
    .OUTPUT(y1, TensorType({ DT_FLOAT16 }))
    .OUTPUT(y2, TensorType({ DT_INT32 }))
    .ATTR(axis, Int, -1)
    .ATTR(descending, Bool, false)
    .OP_END_FACTORY_REG(Sort)
}  // namespace ge

#endif  // GE_OP_Sort_H