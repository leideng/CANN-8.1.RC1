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
 * \file top_k.cc
 * \brief
 */
#include "top_k.h"

#include "util/util.h"

namespace ge {
static void TopKGetShapeRange(std::vector<std::pair<int64_t, int64_t>> &shape_range,
                              const std::vector<int64_t> &dims_in, int64_t k,
                              uint32_t sorted_axis) {
  for (size_t i = 0; i < dims_in.size(); i++) {
    if (i == sorted_axis && k > 0) {
      shape_range.push_back(pair<int64_t, int64_t>(k, k));
    } else if (dims_in[i] == UNKNOWN_DIM) {
      shape_range.push_back(pair<int64_t, int64_t>(1, -1));
    } else {
      shape_range.push_back(pair<int64_t, int64_t>(dims_in[i], dims_in[i]));
    }
  }
}

static bool TopKInferCommon(Operator &op, int64_t k) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto output_v_desc = op_info->MutableOutputDesc("values");
  auto output_i_desc = op_info->MutableOutputDesc("indices");

  std::vector<int64_t> dims_in = input_desc->MutableShape().GetDims();
  int32_t dim_size = dims_in.size();
  if (dim_size <= 0) {
    OP_LOGE(op.GetName().c_str(), "The dims_in size should more than 0!");
    return false;
  }

  int32_t dim = dim_size - 1;
  int32_t sorted_axis = dim;
  if (op.GetAttr("dim", dim) == GRAPH_SUCCESS) {
    sorted_axis = dim;
    if (sorted_axis < 0) {
      sorted_axis += dim_size;
    }
    if (sorted_axis >= dim_size) {
      OP_LOGE(op.GetName().c_str(), "Dim is out of shape size.");
      return false;
    }
  }
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  input_desc->GetShapeRange(shape_range);
  if (shape_range.size() > 0) {
    if (k > 0 && static_cast<int64_t>(sorted_axis) < static_cast<int64_t>(shape_range.size())) {
      shape_range[sorted_axis].first = k;
      shape_range[sorted_axis].second = k;
    }
  } else {
    // input is static shape
    TopKGetShapeRange(shape_range, dims_in, k, static_cast<uint32_t>(sorted_axis));
  }

  bool unknown_rank = IsUnknownRankShape(dims_in);
  if (unknown_rank) {
    output_v_desc->SetShape(GeShape(UNKNOWN_RANK));
    output_v_desc->SetOriginShape(GeShape(UNKNOWN_RANK));

    output_i_desc->SetShape(GeShape(UNKNOWN_RANK));
    output_i_desc->SetOriginShape(GeShape(UNKNOWN_RANK));
  } else {
    dims_in[sorted_axis] = k;

    output_v_desc->SetShape(GeShape(dims_in));
    output_v_desc->SetShapeRange(shape_range);

    output_i_desc->SetShape(GeShape(dims_in));
    output_i_desc->SetShapeRange(shape_range);
  }
  output_v_desc->SetDataType(input_desc->GetDataType());
  output_i_desc->SetDataType(DT_INT32);
  return true;
}
// ----------------TopK Op-------------------
IMPLEMT_VERIFIER(TopK, TopKVerify) { return GRAPH_SUCCESS; }

IMPLEMT_COMMON_INFERFUNC(TopKInferShape) {
  const vector<string> depend_names = {"k"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  auto op_info = OpDescUtils::GetOpDescFromOperator(op);

  Tensor k_tensor;
  bool unkonwn_dim_flag{false};
  if (op.GetInputConstData("k", k_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constdata failed, unknown dim.");
    unkonwn_dim_flag = true;
  }

  // Tensor::GetData() return a uint8 ptr. However the definition of k is int32.
  // So here use int32* ptr to get the k value
  int64_t k = UNKNOWN_DIM;
  if (!unkonwn_dim_flag && k_tensor.GetData() != nullptr) {
    DataType dtype = op.GetInputDesc("k").GetDataType();
    if (dtype == DT_INT32) {
      k = static_cast<int64_t>(*(reinterpret_cast<int32_t*>(k_tensor.GetData())));
    } else if (dtype == DT_INT64) {
      k = *(reinterpret_cast<int64_t*>(k_tensor.GetData()));
    } else {
      OP_LOGE(op.GetName().c_str(), "The type of k Error!");
      return GRAPH_FAILED;
    }
  }

  if (TopKInferCommon(op, k) == false) {
    OP_LOGE(op.GetName().c_str(), "TopKInferCommon Failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TopK, TopKInferShape);
VERIFY_FUNC_REG(TopK, TopKVerify);
// ----------------TopK Op End-------------------
}  // namespace ge
