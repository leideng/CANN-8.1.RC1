/*
 * Copyright (C)  2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file slice.cc
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "./slice.h"

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

namespace ge {
    // ----------------Slice Op Begin ----------------------
    static void GetSliceConstValue(const Tensor& const_tensor, const DataType& dtype, std::vector<int64_t>& const_data) {
      size_t size = 0;
      if (dtype == ge::DT_INT32) {
        int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
        size = const_tensor.GetSize() / sizeof(int32_t);
        for (size_t i = 0; i < size; ++i) {
          const_data.push_back((int32_t)((*(const_data_ptr + i))));
        }
      } else {
        int64_t* const_data_ptr = (int64_t*)const_tensor.GetData();
        size = const_tensor.GetSize() / sizeof(int64_t);
        for (size_t i = 0; i < size; ++i) {
          const_data.push_back(((int64_t)(*(const_data_ptr + i))));
        }
      }
    }

    IMPLEMT_COMMON_INFERFUNC(SliceInferShape) {
      const vector<string> depend_names = {"offsets", "size"};
      PREPARE_DYNAMIC_SHAPE(depend_names);

      Tensor input_begin_tensor;
      Tensor input_size_tensor;
      auto input_desc = op.GetInputDesc("x");
      const Shape shape = input_desc.GetShape();
      DataType input_dtype = input_desc.GetDataType();
      std::vector<int64_t> input_begin;
      std::vector<int64_t> input_size;

      bool has_offsets = true;
      if (op.GetInputConstData("offsets", input_begin_tensor) != GRAPH_SUCCESS) {
        OP_LOGI(op.GetName().c_str(), "Get offsets failed.");
        has_offsets = false;
      } else {
        DataType input_begin_dtype = op.GetInputDesc("offsets").GetDataType();
        GetSliceConstValue(input_begin_tensor, input_begin_dtype, input_begin);
      }

      bool has_size = true;
      if (op.GetInputConstData("size", input_size_tensor) != GRAPH_SUCCESS) {
        OP_LOGI(op.GetName().c_str(), "Get size failed.");
        has_size = false;
      } else {
        DataType input_size_dtype = op.GetInputDesc("size").GetDataType();
        GetSliceConstValue(input_size_tensor, input_size_dtype, input_size);
      }

      bool is_unknown_rank = !has_size && !has_offsets && shape.GetDims() == UNKNOWN_RANK;
      if (is_unknown_rank) {
        TensorDesc output_desc = op.GetOutputDesc("y");
        output_desc.SetDataType(input_dtype);
        Shape outputShape(UNKNOWN_RANK);
        output_desc.SetShape(outputShape);
        OP_LOGD(op.GetName().c_str(), "output_shape:%s", to_string(output_desc.GetShape()).c_str());
        (void) op.UpdateOutputDesc("y", output_desc);
        return GRAPH_SUCCESS;
      }

      auto shape_dims = shape.GetDims();
      if (shape.GetDims() == UNKNOWN_RANK) {
        shape_dims.assign(std::max(input_begin.size(), input_size.size()), -1);
      }

      size_t dimNum = shape_dims.size();
      std::vector<int64_t> outputList;

      vector<pair<int64_t, int64_t>> ranges;
      input_desc.GetShapeRange(ranges);
      if (ranges.empty()) {
        MakeUpShapeRange(shape_dims, ranges);
      }

      if (!has_size && !has_offsets) {
        for (size_t i = 0; i < dimNum; ++i) {
          outputList.push_back(-1);
          ranges[i].first = 1;
        }
      } else if (!has_offsets && has_size) {
        for (size_t i = 0; i < dimNum; ++i) {
          if (input_size[i] == -1) {
            outputList.push_back(-1);
            ranges[i].first = 1;
          } else {
            outputList.push_back(input_size[i]);
            ranges[i].first = input_size[i];
            ranges[i].second = input_size[i];
          }
        }
      } else if (has_offsets && !has_size) {
        for (size_t i = 0; i < dimNum; ++i) {
          outputList.push_back(-1);
          ranges[i].first = 1;
          if (ranges[i].second != -1) {
            if (shape_dims[i] != -1) {
              ranges[i].second = std::min(ranges[i].second, shape_dims[i]);
            }
            ranges[i].second -= input_begin[i];
          }
        }
      } else {
        for (size_t i = 0; i < dimNum; ++i) {
          if (input_size[i] == -1) {
            if (shape_dims[i] == -1) {
              outputList.push_back(-1);
            } else {
              outputList.push_back(shape_dims[i] - input_begin[i]);
            }

            ranges[i].first = 1;
          } else {
            outputList.push_back(input_size[i]);
            ranges[i].first = input_size[i];
            ranges[i].second = input_size[i];
          }
        }
      }

      TensorDesc tensordesc_output = op.GetOutputDesc("y");
      tensordesc_output.SetDataType(input_dtype);
      if (IsUnKnownShape(outputList)) {
        tensordesc_output.SetShapeRange(ranges);
        OP_LOGD(op.GetName().c_str(), "output_ranges:%s", to_string(ranges).c_str());
      }

      Shape outputShape(outputList);
      tensordesc_output.SetShape(outputShape);
      OP_LOGD(op.GetName().c_str(), "output_ranges:%s", to_string(ranges).c_str());
      OP_LOGD(op.GetName().c_str(), "offset:%s", to_string(input_begin).c_str());
      OP_LOGD(op.GetName().c_str(), "size:%s", to_string(input_size).c_str());
      OP_LOGD(op.GetName().c_str(), "output_shape:%s", to_string(tensordesc_output.GetShape()).c_str());
      (void) op.UpdateOutputDesc("y", tensordesc_output);
      return GRAPH_SUCCESS;
    }

    COMMON_INFER_FUNC_REG(Slice, SliceInferShape);
    // ----------------Slice Op End----------------------
}
