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
 * \file reduce_std.cpp
 * \brief
 */
#include <iostream>
#include <algorithm>
#include "reduce_std.h"
#include "op_log.h"

namespace ge {
// ----------------ReduceStd Begin-------------------
using std::find;
IMPLEMT_INFERFUNC(ReduceStd, ReduceStdInferShape) {
  TensorDesc tensordesc_input = op.GetInputDesc("x");
  Shape input_shape = tensordesc_input.GetShape();
  DataType input_dtype = tensordesc_input.GetDataType();
  std::vector<int64_t> dims_input = input_shape.GetDims();
  int64_t dim_num = input_shape.GetDimNum();

  TensorDesc tensordesc_output1 = op.GetOutputDesc("y1");
  TensorDesc tensordesc_output2 = op.GetOutputDesc("y2");
  tensordesc_output1.SetDataType(input_dtype);
  tensordesc_output2.SetDataType(input_dtype);

  bool keepdim;
  (void)op.GetAttr("keepdim", keepdim);

  // check parameter dim and keepdim
  std::vector<int64_t> axis;
  if (GRAPH_SUCCESS != op.GetAttr("dim", axis)) {
    OP_LOGE(op.GetName().c_str(), "GE get dim failed");
    return GRAPH_FAILED;
  }

  for (int i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) {
      axis[i] = axis[i] + dim_num;
    }
  }

  if (axis.empty()) {
    for (int i = 0; i < dim_num; i++) {
      axis.push_back(i);
    }
  }

  std::vector<int64_t> oshape_vector;
  for (int item = 0; item < dim_num; ++item) {
    if (find(axis.begin(), axis.end(), item) != axis.end()) {
      // item in axis
      if (keepdim == true) {
        // If keepDims is true, current dimesion set to 1
        oshape_vector.push_back(1);
      }
    } else {
      // item is not in ConstValueAxis
      oshape_vector.push_back(dims_input[item]);
    }
  }

  Shape oshape(oshape_vector);
  tensordesc_output1.SetShape(oshape);
  tensordesc_output2.SetShape(oshape);
  (void)op.UpdateOutputDesc("y1", tensordesc_output1);
  (void)op.UpdateOutputDesc("y2", tensordesc_output2);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ReduceStd, ReduceStdInferShape);
// ----------------ReduceStd END---------------------

}  // namespace ge