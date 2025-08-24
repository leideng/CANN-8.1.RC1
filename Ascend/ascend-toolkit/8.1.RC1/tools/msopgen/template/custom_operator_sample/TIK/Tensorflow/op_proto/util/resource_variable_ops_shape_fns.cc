/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file resource_variable_ops_shape_fns.cpp
 * \brief
 */
#include "resource_variable_ops_shape_fns.h"

#include <vector>

#include "graph/types.h"
#include "op_log.h"
#include "common_shape_fns.h"
#include "error_util.h"

namespace ge {

static graphStatus ValidateVariableResourceHandle(Operator& op, std::vector<ShapeAndType>& shape_and_type) {
  auto input_handle = op.GetInferenceContext()->GetInputHandleShapesAndTypes();
  if (input_handle.empty()) {
    Shape unknown_shape(ge::UNKNOWN_SHAPE);
    ShapeAndType shape_and_type(unknown_shape, DT_UNDEFINED);
    std::vector<ShapeAndType> handle_shapes_and_types;
    handle_shapes_and_types.reserve(1);
    handle_shapes_and_types.emplace_back(shape_and_type);
    input_handle.emplace_back(handle_shapes_and_types);
  } else {
    shape_and_type = input_handle[0];
    DataType value_type;
    if (op.GetAttr("dtype", value_type) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
          string("get attr[dtype] failed."));
      return GRAPH_FAILED;
    }
    if (shape_and_type[0].GetDataType() != value_type) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
          ConcatString("data type from shape and type context and that from "
              "attr[dtype] do not match, ",
              DTypeStr(shape_and_type[0].GetDataType()), " and ",
              DTypeStr(value_type)));
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus CreateAssignShapeFn(Operator& op) {
  std::vector<ShapeAndType> shape_and_type;
  if (ValidateVariableResourceHandle(op, shape_and_type) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  Shape shape = op.GetInputDesc(1).GetShape();
  Shape unused;
  if (Merge(shape_and_type[0].GetShape(), shape, unused, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        ConcatString("failed to call Merge function to merge 1th input shape",
            DebugString(shape.GetDims()), " and shape",
            DebugString(shape_and_type[0].GetShape().GetDims()),
            " from shape and type context"));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
}  // namespace ge
