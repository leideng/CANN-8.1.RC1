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

/*!
 * \file nn_pooling_ops.cpp
 * \brief
 */
/* reslove the complexity of pooling fuction. */
#include "adaptive_max_pool2d.h"
#include "op_log.h"

namespace ge {
// ------------AdaptiveMaxPool2d Op Begin----------------
IMPLEMT_INFERFUNC(AdaptiveMaxPool2d, AdaptiveMaxPool2dInferShape) {
  OP_LOGI(op.GetName().c_str(), " AdaptiveMaxPool2d inferShape begin!");
  const size_t DIM_SIZE2 = 2;
  auto input_tensor_desc = op.GetInputDesc("x");
  auto shape = input_tensor_desc.GetShape();
  // get output_size
  std::vector<int64_t> ouput_size_list;
  if (GRAPH_SUCCESS != op.GetAttr("output_size", ouput_size_list)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ouput_size_list failed!");
    return GRAPH_FAILED;
  }
  // check output size
  if (ouput_size_list.size() != DIM_SIZE2) {
    OP_LOGE(op.GetName().c_str(), "length of output_size must be 2");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims_input = shape.GetDims();
  // set output shape
  std::vector<int64_t> dim_vector;
  for (size_t i = 0; i < dims_input.size(); i++) {
    int64_t dims = dims_input[i];
    dim_vector.push_back(dims);
  }
  OP_LOGI(op.GetName().c_str(),
          " AdaptiveMaxPool2d inferShape dims: [%u] ", dims_input.size());

  size_t index0 = dims_input.size() - 2;
  size_t index1 = dims_input.size() - 1;
  Format input_format = input_tensor_desc.GetFormat();
  OP_LOGI(op.GetName().c_str(), " AdaptiveMaxPool2d inferShape format: [%u] ", input_format);
  if ((input_format == FORMAT_NC1HWC0) || (input_format == FORMAT_NHWC)) {
    index0 = dims_input.size() - 3;
    index1 = dims_input.size() - 2;
  }
  dim_vector[index0] = ouput_size_list[0];
  dim_vector[index1] = ouput_size_list[1];

  TensorDesc td = op.GetOutputDesc("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  Shape output_shape(dim_vector);
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);

  TensorDesc out1_td = op.GetOutputDesc("argmax");
  DataType out1_dtype = out1_td.GetDataType();
  OP_LOGI(op.GetName().c_str(),  " AdaptiveMaxPool2d inferShape argmax dtype: [%u] ", out1_dtype);
  if (out1_dtype == DT_UNDEFINED) {
      out1_td.SetDataType(DT_INT32);
      OP_LOGI(op.GetName().c_str(), " AdaptiveMaxPool2d inferShape set argmax dtype: [%u] ", DT_INT32);
  }
  out1_td.SetShape(output_shape);
  (void)op.UpdateOutputDesc("argmax", out1_td);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(AdaptiveMaxPool2d, AdaptiveMaxPool2dVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdaptiveMaxPool2d, AdaptiveMaxPool2dInferShape);
VERIFY_FUNC_REG(AdaptiveMaxPool2d, AdaptiveMaxPool2dVerify);
// ------------AdaptiveMaxPool2d Op End----------------
}  // namespace ge