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
 * \file space_to_depth.cpp
 * \brief
 */
#ifdef CHECK_FORMAT
#undef CHECK_FORMAT
#endif

#define CHECK_FORMAT(format)                                                     \
  {                                                                              \
    if (ge::FORMAT_RESERVED == format) {                                         \
      OP_LOGE(op.GetName().c_str(), "get format failed:%s:%d", #format, format); \
      return GRAPH_FAILED;                                                       \
    }                                                                            \
  }

#include "space_to_depth.h"
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "util/util.h"
#include "common_shape_fns.h"
#include "op_log.h"
#include "error_util.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"

  // namespace ge
namespace ge {
// ----------------SpaceToDepth Op Start-------------------
IMPLEMT_VERIFIER(SpaceToDepth, SpaceToDepthVerify) {
  // verify input shape size
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  if (!IsUnknownRankShape(input_dims) && (input_dims.size() < 4)) {
    string excepted_value = ConcatString("greater than or equal to 4.");
    std::string err_msg = GetAttrSizeErrMsg("Input shape", ConcatString(input_dims.size()), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // verify block size
  int64_t block_size;
  if (op.GetAttr("block_size", block_size) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("block_size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (block_size < 2) {
    string excepted_value = ConcatString("greater than or equal to 2");
    std::string err_msg = GetAttrValueErrMsg("block_size", ConcatString(block_size), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // verify data_format
  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format != "NHWC" && data_format != "NCHW" && data_format != "NC1HWC0") {
    string expected_format_list = ConcatString("NHWC, NCHW, NC1HWC0");
    std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SpaceToDepthInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();
  auto input_format = input_desc->GetFormat();

  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  // get attr block_size
  int64_t block_size;
  if (GRAPH_SUCCESS != op.GetAttr("block_size", block_size)) {
    std::string err_msg = GetInputInvalidErrMsg("block_size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // not dynamic case, only set shape
  if (!IsUnknown(input_dims)) {
    std::vector<int64_t> output_dims;
    output_dims.push_back(input_dims[0]);
    if (input_format == FORMAT_NCHW) {
      output_dims.push_back(input_dims[1] * block_size * block_size);
      output_dims.push_back(input_dims[2] / block_size);
      output_dims.push_back(input_dims[3] / block_size);
    } else { // without NCHW all other format set as NHWC
      output_dims.push_back(input_dims[1] / block_size);
      output_dims.push_back(input_dims[2] / block_size);
      output_dims.push_back(input_dims[3] * block_size * block_size);
    }
    output_desc->SetShape(GeShape(output_dims));
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -2, output is -2
  if (IsUnknownRankShape(input_dims)) {
    output_desc->SetShape(GeShape(input_dims));
    OP_LOGW(op.GetName().c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -1, output is -1
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_dims, input_range);

  // infer output shape and range
  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  output_dims.push_back(input_dims[0]);
  output_range.push_back(input_range[0]);
  int64_t dim;
  int64_t range_min;
  int64_t range_max;
  if (input_format == FORMAT_NCHW) {
    dim = input_dims[1] == -1 ? -1 : input_dims[1] * block_size * block_size;
    range_min = input_range[1].first * block_size * block_size;
    range_max = input_range[1].second == -1 ? -1 : input_range[1].second * block_size * block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[2] == -1 ? -1 : input_dims[2] / block_size;
    range_min = input_range[2].first / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[2].second == -1 ? -1 : input_range[2].second / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[3] == -1 ? -1 : input_dims[3] / block_size;
    range_min = input_range[3].first / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[3].second == -1 ? -1 : input_range[3].second / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  } else {
    dim = input_dims[1] == -1 ? -1 : input_dims[1] / block_size;
    range_min = input_range[1].first / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[1].second == -1 ? -1 : input_range[1].second / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[2] == -1 ? -1 : input_dims[2] / block_size;
    range_min = input_range[2].first / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[2].second == -1 ? -1 : input_range[2].second / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[3] == -1 ? -1 : input_dims[3] * block_size * block_size;
    range_min = input_range[3].first * block_size * block_size;
    range_max = input_range[3].second == -1 ? -1 : input_range[3].second * block_size * block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  }

  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SpaceToDepth, SpaceToDepthInferShape);
VERIFY_FUNC_REG(SpaceToDepth, SpaceToDepthVerify);
// ----------------SpaceToDepth Op End-------------------

}  // namespace ge