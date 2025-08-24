/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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


[Introduction]
This sample contains code samples for TensorFlow custom operator development and provides corresponding build scripts.
Developers can add their own custom operator implementations based on this sample and then build the project to obtain a custom operator package (OPP).


[Sample Overview]
*****custom operator samples*****
ApplyAdamD
Function: Updates "var" according to the Adam algorithm.

BasicLSTMCell
Function: Basic LSTM Cell forward calculation.

BatchNorm
Function: Perform data normalization on FeatureMap.

Conv3D
Function: Computes a 3D convolution given 5D "x" and "filter" tensors.

Fill
Function: Creates a tensor filled with a scalar value.


MaxPool
Function: Perform max pooling on the input.

Mul
Function: Return x1 * x2 element-wise.


ReduceAll
Function: Calculate the "logical sum" of elements of a tensor in a dimension.

Conv2D
Function: Computes a 2D convolution given 4D "x" and "filter" tensors.

Relu
Function: Computes rectified linear: "max(x, 0)".
