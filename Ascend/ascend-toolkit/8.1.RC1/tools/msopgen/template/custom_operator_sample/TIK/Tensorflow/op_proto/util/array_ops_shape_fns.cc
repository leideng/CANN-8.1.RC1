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
 * \file array_ops_shape_fns.cpp
 * \brief
 */
#include "array_ops_shape_fns.h"
#include "graph/types.h"
#include "op_log.h"
#include "error_util.h"
#include "common_shape_fns.h"
#include "graph/utils/op_desc_utils.h"
#include "axis_util.h"
#include "util.h"

namespace ge {
static graphStatus CalcPadGradOutDims(const Shape& input_shape, const Tensor& paddings_tensor,
                                      std::vector<int64_t>& output_dims, const ge::Operator& op) {
  graphStatus status;
  size_t input_rank = input_shape.GetDimNum();
  if (output_dims.size() < input_rank) {
    return GRAPH_FAILED;
  }
  DataType padding_type = paddings_tensor.GetTensorDesc().GetDataType();
  if (padding_type == DT_INT32) {
    const int32_t* paddings_data = reinterpret_cast<const int32_t*>(paddings_tensor.GetData());
    CHECK(paddings_tensor.GetSize() / sizeof(int32_t) < input_rank,
          OP_LOGE(op, "invalid padding data."), return GRAPH_FAILED);
    for (size_t i = 0; i < input_rank; ++i) {
      const int64_t pad0 = static_cast<int64_t>(paddings_data[2 * i]);
      const int64_t pad1 = static_cast<int64_t>(paddings_data[(2 * i) + 1]);
      if ((pad0 < 0) || (pad1 < 0)) {
        OP_LOGE(op, "Paddings must be non-negative, pad0= %ld, pad1=%ld.", pad0, pad1);
        return GRAPH_FAILED;
      }
      status = Subtract(input_shape.GetDim(i), pad0 + pad1, output_dims[i], op);
      if (status != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    }
  } else if (padding_type == DT_INT64) {
    const int64_t* paddings_data = reinterpret_cast<const int64_t*>(paddings_tensor.GetData());
    CHECK(paddings_tensor.GetSize() / sizeof(int64_t) < input_rank,
          OP_LOGE(op, "invalid padding data."), return GRAPH_FAILED);
    for (size_t i = 0; i < input_rank; ++i) {
      const int64_t pad0 = paddings_data[2 * i];
      const int64_t pad1 = paddings_data[(2 * i) + 1];
      if ((pad0 < 0) || (pad1 < 0)) {
        OP_LOGE(op, "Paddings must be non-negative, pad0=%ld, pad1=%ld.", pad0, pad1);
        return GRAPH_FAILED;
      }
      status = Subtract(input_shape.GetDim(i), pad0 + pad1, output_dims[i], op);
      if (status != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    }
  } else {
    OP_LOGE(op, "Data type invalid, should be DT_INT32 or DT_INT64");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus PadGradShapeFn(Operator& op) {
  const vector<string> depend_names = {"paddings"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  Shape paddings;
  graphStatus status = WithRank(op.GetInputDesc(1), 2, paddings, op);
  if (status != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        ConcatString("call WithRank failed, ", GetShapeErrMsg(1,
            DebugString(op.GetInputDesc(1).GetShape().GetDims()), "2D")));
    return GRAPH_FAILED;
  }
  int64_t input_rank = paddings.GetDim(0);
  TensorDesc output_desc = op.GetOutputDescByName("y");
  output_desc.SetDataType(op.GetInputDesc(0).GetDataType());
  if (input_rank == UNKNOWN_DIM) {
    OP_LOGE(TbeGetName(op).c_str(), "paddings inputShape of 0 dims is unknown, set out shape unknown.");
    output_desc.SetShape(Shape(UNKNOWN_SHAPE));
    return op.UpdateOutputDesc("y", output_desc);
  }

  Shape input_shape;
  if (WithRank(op.GetInputDesc(0), input_rank, input_shape, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        ConcatString("call WithRank failed, ", GetShapeErrMsg(0,
            DebugString(op.GetInputDesc(0).GetShape().GetDims()), ConcatString(input_rank))));
    return GRAPH_FAILED;
  }

  Shape check_shape({input_rank, 2});
  if (Merge(paddings, check_shape, paddings, op)) {
    string err_msg = ConcatString("merge 1th input shape", DebugString(paddings.GetDims()), " and shape",
                                  DebugString(check_shape.GetDims()), " failed");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  size_t dim_num = op.GetInputDescByName("x").GetShape().GetDimNum();
  std::vector<int64_t> empty_dim_vec = op.GetInputDescByName("x").GetShape().GetDims();
  for (size_t i = 0; i < dim_num; i++) {
    if (empty_dim_vec[i] == 0) {
      output_desc.SetShape(Shape(empty_dim_vec));
      return op.UpdateOutputDesc("y", output_desc);
    }
  }

  Tensor paddings_tensor;
  if (op.GetInputConstData("paddings", paddings_tensor) != GRAPH_SUCCESS) {
    std::vector<int64_t> unknow_dim_vec(input_rank, UNKNOWN_DIM);
    OP_LOGD(TbeGetName(op).c_str(), "Get paddings input tensor fail, set outPut shape unknown.");
    output_desc.SetShape(Shape(unknow_dim_vec));
    return op.UpdateOutputDesc("y", output_desc);
  }

  std::vector<int64_t> output_dims(input_rank);
  auto result = CalcPadGradOutDims(input_shape, paddings_tensor, output_dims, op);
  if (result != GRAPH_SUCCESS) {
    string err_msg = ConcatString("calculate out dims failed,", "please check the validity of input and attribute");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  output_desc.SetShape(Shape(output_dims));
  return op.UpdateOutputDesc("y", output_desc);
}
}  // namespace ge
