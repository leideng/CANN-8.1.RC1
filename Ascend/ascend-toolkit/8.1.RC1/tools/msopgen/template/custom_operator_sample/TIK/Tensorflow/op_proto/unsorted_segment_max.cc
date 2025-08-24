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
 * \file unsorted_segment_max.cc
 * \brief
 */
#include "unsorted_segment_max.h"

#include <cmath>
#include <string>
#include <vector>

#include "util/util.h"
#include "error_util.h"
#include "op_log.h"
#include "graph/utils/op_desc_utils.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "axis_util.h"
#include "common_shape_fns.h"

#define ELLIPSIS_MASK_UPDATE(mask, new_mask, bit_ellipsis, i, pow_table, \
                             right_mov)                                  \
  do {                                                                   \
    if (((mask) & (1 << i)) && (bit_ellipsis >= i)) {                    \
      new_mask += pow_table[i];                                          \
    } else if (((mask) & (1 << i)) && (bit_ellipsis < i)) {              \
      new_mask += pow_table[i + right_mov];                              \
    }                                                                    \
  } while (0)

namespace ge {
static bool CheckListEmpty(const std::string& op_name, const std::vector<int64_t>& list, const std::string& attr_name) {
  if (list.empty()) {
    OP_LOGE(op_name.c_str(), "The %s is empty !", attr_name.c_str());
    return false;
  }
  return true;
}
static std::vector<int64_t> GetAttrValue(const ge::Operator& op, const std::string& key_name) {
  std::vector<int64_t> list;
  if (op.GetAttr(key_name, list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue failed!");
  }
  return list;
}
static void GetUnsortedSegmentSumConstValue(const Tensor& const_tensor, const DataType& dtype, int64_t& const_data) {
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
    const_data = (int32_t)((*(const_data_ptr + 0)));
  } else {
    int64_t* const_data_ptr = (int64_t*)const_tensor.GetData();
    const_data = (int64_t)(*(const_data_ptr + 0));
  }
}

IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentSumInferShape) {
  vector<string> input_infer_depends = {"num_segments"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);

  Tensor input_num_segments_tensor;
  int64_t input_num_segments;
  DataType input_num_segments_dtype = op_desc->MutableInputDesc("num_segments")->GetDataType();

  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op_desc->MutableInputDesc("x")->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_seg_id;
  op_desc->MutableInputDesc("segment_ids")->GetShapeRange(shape_range_seg_id);

  std::vector<std::pair<int64_t, int64_t>> out_range;

  if (GRAPH_SUCCESS != op.GetInputConstData("num_segments", input_num_segments_tensor)) {
    input_num_segments = -1;
    out_range.push_back(std::pair<int64_t, int64_t>(1, -1));
  } else {
    GetUnsortedSegmentSumConstValue(input_num_segments_tensor, input_num_segments_dtype, input_num_segments);
    out_range.push_back(std::pair<int64_t, int64_t>(input_num_segments, input_num_segments));
  }

  ge::GeShape shape = op_desc->MutableInputDesc("x")->GetShape();
  ge::GeShape shape_id = op_desc->MutableInputDesc("segment_ids")->GetShape();
  auto shape_vec = shape.GetDims();
  auto shape_id_vec = shape_id.GetDims();

  MakeUpShapeRange(shape_vec, shape_range_x);
  MakeUpShapeRange(shape_id_vec, shape_range_seg_id);

  int64_t dim_idsize_input = shape_id.GetDimNum();
  int64_t dim_size_input = shape.GetDimNum();
  DataType input_dtype = op_desc->MutableInputDesc("x")->GetDataType();
  vector<int64_t> shape_vector;
  if (IsUnknownRankShape(shape_vec) || IsUnknownRankShape(shape_id_vec)) {
    shape_vector.push_back(-2);
    for (size_t i = shape_range_seg_id.size(); i < shape_range_x.size(); i++) {
      out_range.push_back(shape_range_x[i]);
    }
  } else if (dim_idsize_input > 1) {
    shape_vector.push_back(input_num_segments);
    for (int i = dim_idsize_input; i < dim_size_input; i++) {
      shape_vector.push_back(shape_vec[i]);
      if ((size_t)i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
    }
  } else {
    shape_vector = shape_vec;
    shape_vector[0] = input_num_segments;
    for (size_t i = 1; i < shape_vector.size(); i++) {
      if (i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
    }
  }

  GeTensorDescPtr tensordesc_output = op_desc->MutableOutputDesc("y");
  ge::GeShape out_shape = ge::GeShape(shape_vector);
  tensordesc_output->SetShape(out_shape);
  tensordesc_output->SetDataType(input_dtype);
  tensordesc_output->SetShapeRange(out_range);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(UnsortedSegmentMax, UnsortedSegmentSumInferShape);
}  // namespace ge