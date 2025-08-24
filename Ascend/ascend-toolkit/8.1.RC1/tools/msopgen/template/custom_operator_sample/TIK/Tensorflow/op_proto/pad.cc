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
 * \file pad.cpp
 * \brief
 */
#include "pad.h"

#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#include "graph/utils/node_utils.h"

#include "util/util.h"
#include "util/common_shape_fns.h"
#include "error_util.h"
#include "op_log.h"

namespace ge {
// ----------------Pad Op Begin-------------------
static graphStatus PadInferShapeAndType(ge::Operator& op, std::vector<int64_t>& paddings) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_shape = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();
  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  if (!IsUnknown(input_shape)) {
    // not dynamic shape, will output shape and dtype
    if (input_shape.empty()) {
      input_shape.push_back(1);
    }
    if (input_shape.size() * 2 != paddings.size()) {
      OP_LOGE("OP[Pad]", "the num of paddings must be double the input dim size");
      return GRAPH_FAILED;
    }

    // calce the output shape
    vector<int64_t> output_shape;
    for (size_t dim = 0; dim < input_shape.size(); dim++) {
      output_shape.push_back(input_shape[dim] + paddings[dim * 2] + paddings[dim * 2 + 1]);
    }
    output_desc->SetShape(GeShape(output_shape));
  
    return GRAPH_SUCCESS;
  }

  // input shape is -2, output is -2
  if (IsUnknownRankShape(input_shape)) {
    output_desc->SetShape(GeShape(input_shape));
  
    return GRAPH_SUCCESS;
  }

  // input shape is -1, will get the shape and range
  // calcu the output shape
  vector<int64_t> output_shape;
  for (size_t dim = 0; dim < input_shape.size(); dim++) {
    if (input_shape[dim] == -1) {
      output_shape.push_back(input_shape[dim]);
    } else {
      output_shape.push_back(input_shape[dim] + paddings[dim * 2] + paddings[dim * 2 + 1]);
    }
  }
  output_desc->SetShape(GeShape(output_shape));

  // calcu the output range
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_shape, input_range);
  std::vector<std::pair<int64_t, int64_t>> output_range;
  for (size_t dim = 0; dim < input_shape.size(); dim++) {
    auto range_min = input_range[dim].first + paddings[dim * 2] + paddings[dim * 2 + 1];
    auto range_max = input_range[dim].second == -1 ?
                     -1 : input_range[dim].second + paddings[dim * 2] + paddings[dim * 2 + 1];
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  }
  output_desc->SetShapeRange(output_range);

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(PadInferShape) {
  OP_LOGD("OP[Pad]", "PadInferShape Begin.");
  const vector<string> depend_names = {"paddings"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  // first get the padding const
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto pad_idx = static_cast<uint32_t>(op_desc->GetInputIndexByName("paddings"));
  const GeTensor *paddings_tensor = OpDescUtils::GetInputConstData(op, pad_idx);
  if (paddings_tensor != nullptr) {
    OP_LOGW("OP[Pad]", "the node paddings is not const node, will set the output dynamic");
    auto input_desc = op_info->MutableInputDesc("x");
    auto input_shape = input_desc->MutableShape().GetDims();
    DataType input_dtype = input_desc->GetDataType();
    auto output_desc = op_info->MutableOutputDesc("y");
  
    // shape_x is UNKNOWN_RANK
    if (IsUnknownRankShape(input_shape)) {
      OP_LOGW("OP[Pad]", "shape_x is UNKNOWN_RANK. Set output UNKNOWN_RANK");
      output_desc->SetShape(GeShape(input_shape));
      output_desc->SetDataType(input_dtype);
      return GRAPH_SUCCESS;
    }
    // shape_x is UNKNOWN_DIM
    if (input_shape.empty()) {
      input_shape.push_back(-1);
    }
    vector<int64_t> out_shape;
    for (size_t dim = 0; dim < input_shape.size(); dim++) {
      out_shape.push_back(-1);
    }
    std::vector<std::pair<int64_t, int64_t>> output_range;
    MakeUpShapeRange(out_shape, output_range);
    output_desc->SetShape(GeShape(out_shape));
    output_desc->SetDataType(input_dtype);
    output_desc->SetShapeRange(output_range);
    return GRAPH_SUCCESS;
  }

  // get const paddings data
  auto const_desc = op_info->MutableInputDesc("paddings");
  auto const_dtype = const_desc->GetDataType();
  std::vector<int64_t> paddings;
  if (!GetConstValue(op, paddings_tensor, const_dtype, paddings)) {
    OP_LOGE(op.GetName().c_str(), "Get Const paddings value failed, infershape failed");
    return GRAPH_FAILED;
  }

  return PadInferShapeAndType(op, paddings);
}

COMMON_INFER_FUNC_REG(Pad, PadInferShape);
// ----------------Pad Op End-------------------
}  // namespace ge