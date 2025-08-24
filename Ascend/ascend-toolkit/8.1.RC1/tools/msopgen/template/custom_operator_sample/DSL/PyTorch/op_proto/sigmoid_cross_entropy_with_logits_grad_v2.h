/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file sigmoid_cross_entropy_with_logits_grad_v2.h
 * \brief
 */
#ifndef GE_OP_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_H
#define GE_OP_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes gradients of sigmoid_cross_entropy_with_logits_v2.

* @par Inputs:
* @li predict: An ND tensor of type float16, float32.
* @li target: An ND tensor of type float16, float32.
* @li dout: An ND tensor of type float16, float32.
* @li weight: An optional ND tensor of type float16, float32.
* @li pos_weight: An optional ND tensor of type float16, float32. \n

* @par Attributes:
* @li reduction: An optional string.Defaults to "mean". \n

* @par Outputs:
* @li gradient: An ND tensor tensor with the same shape and type as "predict". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator SigmoidCrossEntropyWithLogitsGrad.
*/
REG_OP(SigmoidCrossEntropyWithLogitsGradV2)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(pos_weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogitsGradV2)
}  // namespace ge

#endif  // GE_OP_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_H