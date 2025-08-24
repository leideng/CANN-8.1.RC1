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
 * \file sparse_to_dense.cc
 * \brief
 */
#include "sparse_to_dense.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "graph/utils/op_desc_utils.h"
#include "error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(SparseToDense, SparseToDenseInfer) {
  GeShape shape;
  if (MakeShapeFromShapeTensor(op, "output_shape", shape) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call MakeShapeFromShapeTensor function to make shape from "
        "input[output_shape]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  DataType values_type = op.GetInputDesc("values").GetDataType();

  auto output_desc = op_desc->MutableOutputDesc(0);
  output_desc->SetShape(shape);
  output_desc->SetDataType(values_type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseToDense, SparseToDenseInfer);
}  // namespace ge
