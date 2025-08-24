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
 * \file where.cpp
 * \brief
 */
#include "where.h"

#include "op_log.h"
#include "common_shape_fns.h"
#include "error_util.h"
#include "util/util.h"

namespace ge {
IMPLEMT_INFERFUNC(Where, WhereInfer) {
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr x_desc = op_desc->MutableInputDesc(0);

  GeShape x_shape;
  if (WithRankAtLeast(x_desc, 1, x_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "input x must be at least 1D.");
    return GRAPH_FAILED;
  }

  if (WithRankAtMost(x_desc, 5, x_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "input x must be at most 5D.");
    return GRAPH_FAILED;
  }

  GeTensorDescPtr y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetDataType(DT_INT64);

  vector<int64_t> y_shape;
  auto input_dims = x_shape.GetDims();
  int64_t input_shape_size = x_shape.GetShapeSize();
  if (input_shape_size != UNKNOWN_DIM) {
    // input shape: known
    y_shape.push_back(UNKNOWN_DIM);
    y_shape.push_back(input_dims.size());

    std::vector<std::pair<int64_t, int64_t>> range;
    int64_t dims_num = x_shape.GetDimNum();
    range.emplace_back(std::make_pair(0, input_shape_size));
    range.emplace_back(std::make_pair(dims_num, dims_num));
    y_desc->SetShapeRange(range);
  } else {
    if (input_dims == UNKNOWN_RANK) {
      // input shape: unknown rank
      y_shape.push_back(UNKNOWN_DIM);
      y_shape.push_back(UNKNOWN_DIM);
    } else {
      // input shape: unknown dims
      y_shape.push_back(UNKNOWN_DIM);
      y_shape.push_back(input_dims.size());
    }
  }

  y_desc->SetShape(GeShape(y_shape));
  y_desc->SetOriginShape(GeShape(y_shape));
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Where, WhereInfer);
}  // namespace ge