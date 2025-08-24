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
Add
Function: Returns x + y element-wise.

DecodeBboxV2
Function: Decode the bounding box according to different encoding or decoding methods.

InplaceUpdate
Function: Updates specified rows with values in v.

Pad
Function: Pads a tensor.

ScatterNdAdd
Function: The ScatterNdAdd operator applies the sparse algorithm to a single value or slice in the input data to obtain the output data.

Slice
Function: Extracts a slice from a tensor.

SpaceToDepth
Function: Outputs a copy of the input tensor where values from the "height" and "width" dimensions are moved to the "depth" dimension.

UnsortedSegmentMax
Function: Computes the maximum along segments of a tensor.