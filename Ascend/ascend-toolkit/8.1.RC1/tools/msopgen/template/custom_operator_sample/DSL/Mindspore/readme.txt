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
This sample contains code samples for Mindspore custom operator development.


[Directory Structure]
The directory of a Mindspore custom operator sample project is organized as follows:

Mindspore
    ├── mindspore      // Directory of Mindspore operator implementation and information library files
    │   ├── impl    // Directory of operator implementation files
    │   │      ├── xx.py
    ├── op_proto      // Directory of the operator prototype definition file
    │   ├── xx.py
    ├── testcases     // Directory of the operator test file
    │   ├── readme.txt
    ├── readme.txt


[Sample Overview]
*****custom operator samples*****
Add3
Function: Add all input tensors element-wise.

CorrectionMul
Function: Scale the weights with a correction factor to the long term statistics
prior to quantization. This ensures that there is no jitter in the quantized weights
due to batch to batch variation.

Square
Function: Calculate data's square, y= x * x.