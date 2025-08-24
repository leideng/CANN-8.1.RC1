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
#ifndef GE_OP_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_H
#define GE_OP_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the sigmoid cross entropy loss of "predict" and "target" . \n

* @par Inputs:
* Two inputs, including:
* @li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
* @li target: A multi-dimensional Tensor of type float16 or float32, specifying the target value . \n
* @par Outputs:
* loss: Sigmoid cross entropy between the predictive value and target value. Has the same dimensions as "predict" . \n

* @par Third-party framework compatibility
* Compatible with the scenario where "reduction" is set to "none"of PyTorch operator SigmoidCrossEntropyWithLogitsGrad.
*/
REG_OP(SigmoidCrossEntropyWithLogitsGrad)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogitsGrad)
}  // namespace ge

#endif  // GE_OP_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_H