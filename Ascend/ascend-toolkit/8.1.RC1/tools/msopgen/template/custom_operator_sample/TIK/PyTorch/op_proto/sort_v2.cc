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
 * \file sort_v2.cc
 * \brief
 */
#include "sort_v2.h"


namespace ge {
// ----------------SortV2 Begin-------------------
IMPLEMT_INFERFUNC(SortV2, SortV2InferShape) {
    TensorDesc tensordesc_input = op.GetInputDesc("x");
    Shape input_shape = tensordesc_input.GetShape();
    DataType input_dtype = tensordesc_input.GetDataType();
    std::vector<int64_t> dims_input = input_shape.GetDims();

    TensorDesc tensordesc_output1 = op.GetOutputDesc("y");

    tensordesc_output1.SetShape(ge::Shape(dims_input));

    tensordesc_output1.SetDataType(input_dtype);

    (void)op.UpdateOutputDesc("y", tensordesc_output1);

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SortV2, SortV2Verify) { return GRAPH_SUCCESS; }

INFER_FUNC_REG(SortV2, SortV2InferShape);
VERIFY_FUNC_REG(SortV2, SortV2Verify);
// ----------------SortV2 END---------------------
}  // namespace ge
