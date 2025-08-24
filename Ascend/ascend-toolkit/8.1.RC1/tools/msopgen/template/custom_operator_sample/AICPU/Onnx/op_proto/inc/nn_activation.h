/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file nn_activation.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_ACTIVATION_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_ACTIVATION_H_

#include "graph/operator_reg.h"

namespace ge{
        /**
    * @brief Compute the SwiGlu,
    * where the activations function in GLU is Swish.

    * @par Inputs:
    * One input, including:
    * @x: A Tensor. Must be one of the following types: bfloat16, float16, float32.

    * @par Outputs:
    * one output, including:
    * @y: A Tensor. Must be one of the following types: bfloat16, float16, float32.

    * @par Attributes:
    * two attributes, including:
    * @li dim: A optional int. The dimension to be split, default is -1.

    * @par Third-party framework compatibility:
    * New operator SwiGlu.
    */
    REG_OP(SwiGlu)
        .INPUT(x, "T")
        .OUTPUT(y, "T")
        .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(dim, Int, -1)
        .OP_END_FACTORY_REG(SwiGlu)

    /**
    * @brief Compute the SwiGluGrad,
    * where the activations function in GLU is SwishGrad.

    * @par Inputs:
    * two input, including:
    * @li y_grad: A Tensor, which is the output gradient of forward operator and which \n
    * has the same shape as "x" except for the dimension specified by the "dim" parameter. \n
    * The dimension size specified by "dim" is half of the corresponding dimension of x. \n 
    * Must be one of the following types: bfloat16, float16, float32.
    * @li x: A Tensor. Must be one of the following types: bfloat16, float16, float32.

    * @par Outputs:
    * one Output, including:
    * x_grad: A Tensor, which is the gradient of x and has the same shape as "x". \n
    * Must be one of the following types: bfloat16, float16, float32.

    * @par Attributes:
    * one attributes, including:
    * @li dim: A optional int. The dimension to be split, default is -1.

    * @par Third-party framework compatibility:
    * New operator SwiGluGrad.
    */
    REG_OP(SwiGluGrad)
        .INPUT(y_grad, "T")
        .INPUT(x, "T")
        .OUTPUT(x_grad, "T")
        .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(dim, Int, -1)
        .OP_END_FACTORY_REG(SwiGluGrad)
        
    /**
    * @brief FatreluMul divides the input tensor into left and right tensors x1 and x2 based on the last dimension,
    * performs Threshold calculation on x1 on the left, and multiplies the calculation result by x2. \n

    * @par Inputs:
    * @li x: A tensor of type float, float16 or bfloat16. Shape support 2D ~ 8D.
    * The format must be ND.
    * @li threshold: A scalar, type is float, used to set the threshold of "x".

    * @par Outputs:
    * y: A tensor has the same type and format as "x".
    * Other dimensions of its shape are the same as those of "x".
    * The value of the last dimension is half the value of the last dimension of "x". \n
    */
    REG_OP(FatreluMul)
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .INPUT(threshold, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
        .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .OP_END_FACTORY_REG(FatreluMul)  
    
    /**
    * @brief GeluMul divides the input tensor into left and right tensors x1 and x2 based on the last dimension,
    * performs GELU calculation on x1 on the left, and multiplies the calculation result by x2. \n

    * @par Inputs:
    * x: A tensor of type float, float16 or bfloat16. Shape support 2D ~ 8D.
    * The format must be ND.

    * @par Attributes:
    * approximate: A optional string. The GELU approximation algorithm to use: 'none' or 'tanh', default is 'none'.

    * @par Outputs:
    * y: A tensor has the same type and format as "x".
    * Other dimensions of its shape are the same as those of "x".
    * The value of the last dimension is half the value of the last dimension of "x". \n
    */
    REG_OP(GeluMul)
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(approximate, String, "none")
        .OP_END_FACTORY_REG(GeluMul)
}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_ACTIVATION_H_
