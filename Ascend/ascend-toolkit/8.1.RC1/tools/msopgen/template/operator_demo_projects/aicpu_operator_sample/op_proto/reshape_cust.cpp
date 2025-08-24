/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "reshape_cust.h"
#include <vector>
#include <string>
#include <iostream>

namespace {
template <typename T>
std::vector<int64_t> AsInt64(const T *data, int64_t dataSize)
{
    std::vector<int64_t> ret(dataSize);
    for (int64_t i = 0; i < dataSize; ++i) {
        ret[i] = data[i];
    }
    return ret;
}

int64_t GetElementNum(const std::vector<int64_t> &shape)
{
    int64_t ret = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        ret *= shape[i];
    }
    return ret;
}
}

namespace ge {
IMPLEMT_COMMON_INFERFUNC(ReshapeCustInferShape) {
    TensorDesc tensordesc_tensor = op.GetInputDescByName("tensor");
    TensorDesc tensordesc_shape = op.GetInputDescByName("shape");
    TensorDesc tensordesc_output = op.GetOutputDescByName("output");
    std::vector<AscendString> depends = {"shape"};
    op.SetAttr("_op_infer_depends", depends);
    Tensor shape_tensor;
    if (op.GetInputConstData("shape", shape_tensor) == GRAPH_SUCCESS) {
        DataType shape_type = tensordesc_shape.GetDataType();
        std::vector<int64_t> shape_values;
        if (shape_type == DT_INT32) {
            auto shape_data = reinterpret_cast<const int32_t *>(shape_tensor.GetData());
            shape_values = AsInt64<int32_t>(shape_data, shape_tensor.GetSize() / sizeof(int32_t));
        } else {
            auto shape_data = reinterpret_cast<const int64_t *>(shape_tensor.GetData());
            shape_values = AsInt64<int64_t>(shape_data, shape_tensor.GetSize() / sizeof(int64_t));
        }

        std::vector<int64_t> input_shape = tensordesc_tensor.GetShape().GetDims();
        int64_t input_element_num = GetElementNum(input_shape);
        int64_t shape_element_num = GetElementNum(shape_values);
        if (input_element_num != shape_element_num) {
            return GRAPH_FAILED;
        }
        tensordesc_output.SetShape(Shape(shape_values));
        tensordesc_output.SetOriginShape(Shape(shape_values));
    }

    tensordesc_output.SetDataType(tensordesc_tensor.GetDataType());

    std::vector<std::pair<int64_t, int64_t>> range;
    auto status = op.GetInputDescByName("tensor").GetShapeRange(range);
    if (status != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    tensordesc_output.SetShapeRange(range);

    (void)op.UpdateOutputDesc("output", tensordesc_output);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ReshapeCust, ReshapeCustInferShape);
}
