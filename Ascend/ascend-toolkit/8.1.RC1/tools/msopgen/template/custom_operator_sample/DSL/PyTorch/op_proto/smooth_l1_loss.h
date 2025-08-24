/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef GE_OP_SMOOTHL1LOSS_H
#define GE_OP_SMOOTHL1LOSS_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Computes the regression box of the RPN. It is a FasterRCNN operator . \n

*@par Inputs:
* Two inputs, including:
*@li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
*@li label: A multi-dimensional Tensor of type float16 or float32, specifying the target value . \n

*@par Attributes:
* sigma: Must be a floating point number. Defaults to "1.0" . \n

*@par Outputs:
*loss: Indicates the loss between the predictive value and target value. Has the same dimensions as "predict" . \n

*@attention Constraints:
* This operator does not perform the "reduce" operation on the loss value. Call other reduce operators to perform "reduce" operation on the loss if required . \n

*@par Third-party framework compatibility
* Compatible with the scenario where "reduction" is set to "none"of PyTorch operator SmoothL1Loss.
*/
REG_OP(SmoothL1Loss)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(sigma, Float, 1.0)
    .OP_END_FACTORY_REG(SmoothL1Loss)
}  // namespace ge

#endif  // GE_OP_SMOOTHL1LOSS_H