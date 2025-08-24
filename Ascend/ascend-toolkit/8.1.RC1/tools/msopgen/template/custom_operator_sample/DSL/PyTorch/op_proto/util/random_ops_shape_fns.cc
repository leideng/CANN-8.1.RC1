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
 * \file random_ops_shape_fns.cpp
 * \brief
 */
#include "random_ops_shape_fns.h"
#include "error_util.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"

namespace ge {
graphStatus RandomShape(Operator& op, const std::string& shape_name, const std::string& out_name) {
  std::vector<std::string> input_infer_depends = {shape_name};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);
  Tensor tensor;
  if (op.GetInputConstData(shape_name.c_str(), tensor) != GRAPH_SUCCESS) {
    TensorDesc output_desc = op.GetOutputDescByName(out_name.c_str());
    std::vector<int64_t> vec_shape = op.GetInputDescByName("shape").GetShape().GetDims();
    if (vec_shape[0] >= 0) {
      std::vector<int64_t> tmp_shape(vec_shape[0], ge::UNKNOWN_DIM);
      output_desc.SetShape(ge::Shape(tmp_shape));
    } else {
      output_desc.SetShape(ge::Shape(ge::UNKNOWN_RANK));
    }
    return op.UpdateOutputDesc(out_name.c_str(), output_desc);
  }
  Shape shape;
  if (MakeShapeFromShapeTensor(tensor, shape, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), string("call "
        "MakeShapeFromShapeTensor function failed to get data of shape tensor"));
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDescByName(out_name.c_str());
  output_desc.SetShape(shape);
  return op.UpdateOutputDesc(out_name.c_str(), output_desc);
}

graphStatus RandomShapeWithDataType(Operator& op, const std::string& shape_name, const std::string& date_type_attr_name,
                                    const std::string& out_name) {
  (void)out_name;
  Tensor tensor;
  std::vector<std::string> input_infer_depends = {"shape"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);
  GeTensorDescPtr output_desc = op_desc->MutableOutputDesc(0);
  DataType type;
  if (op.GetAttr(date_type_attr_name.c_str(), type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), ConcatString("get attr[", date_type_attr_name, "] failed"));
    return GRAPH_FAILED;
  }

  if (type != DT_FLOAT16 && type != DT_FLOAT && type != DT_DOUBLE) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       ConcatString(date_type_attr_name, " attr should be half or float32 or double"));
    return GRAPH_FAILED;
  }
  if (op.GetInputConstData(shape_name.c_str(), tensor) != GRAPH_SUCCESS) {
    output_desc->SetDataType(type);
    auto input_shape = op.GetInputDescByName(shape_name.c_str()).GetShape();
    if (input_shape.GetShapeSize() == UNKNOWN_DIM) {
      output_desc->SetShape(GeShape(UNKNOWN_RANK));
      output_desc->SetOriginShape(GeShape(UNKNOWN_RANK));
    } else {
      int64_t rank = input_shape.GetShapeSize();
      std::vector<int64_t> out_shape;
      for (int64_t i = 0; i < rank; i++) {
        out_shape.push_back(UNKNOWN_DIM);
      }
      output_desc->SetShape(GeShape(out_shape));
      output_desc->SetOriginShape(GeShape(out_shape));
    }
    return GRAPH_SUCCESS;
  }
  Shape shape;
  if (MakeShapeFromShapeTensor(tensor, shape, op) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), string("call "
          "MakeShapeFromShapeTensor function failed to get data of shape tensor"));
    return GRAPH_FAILED;
  }
  std::vector<int64_t> output_shape = shape.GetDims();
  output_desc->SetDataType(type);
  output_desc->SetShape(GeShape(output_shape));
  output_desc->SetOriginShape(GeShape(output_shape));
  return GRAPH_SUCCESS;
}
}  // namespace ge
