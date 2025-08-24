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
 * \file candidate_sampling_shape_fns.cpp
 * \brief
 */
#include "candidate_sampling_shape_fns.h"
#include <vector>
#include "op_log.h"
#include "error_util.h"

namespace ge {
constexpr int64_t kRnak = 2;
graphStatus CandidateSamplerShape(Operator& op) {
  int64_t number_true = 0;
  op.GetAttr("num_true", number_true);
  if (number_true < 1) {
    string err_msg = ConcatString("attr[num_true] must >= 1, real value is ",
                                  number_true);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t number_sampled = 0;
  op.GetAttr("num_sampled", number_sampled);
  if (number_sampled < 1) {
    string err_msg = ConcatString("attr[num_sampled] must >= 1, real value is ",
                                  number_sampled);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t range_max = 0;
  op.GetAttr("range_max", range_max);
  if (range_max < 1) {
    string err_msg = ConcatString("attr[range_max] must >= 1, real value is ",
                                  range_max);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape true_classes;
  if (WithRank(op.GetInputDesc(0), kRnak, true_classes, op) != GRAPH_SUCCESS) {
    string err_msg = ConcatString("input[true_classes] must be 2-D, real rank is ",
                                  true_classes.GetDimNum());
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t batch_size = op.GetInputDesc(0).GetShape().GetDim(0);

  std::vector<int64_t> sampled_dimensions;
  sampled_dimensions.reserve(1);
  sampled_dimensions.push_back(number_sampled);

  int64_t true_dims_capacity = 2;
  std::vector<int64_t> true_dimensions;
  true_dimensions.reserve(true_dims_capacity);
  true_dimensions.push_back(batch_size);
  true_dimensions.push_back(number_true);

  TensorDesc candidate_description = op.GetOutputDescByName("sampled_candidates");
  candidate_description.SetShape(Shape(sampled_dimensions));
  candidate_description.SetDataType(DT_INT64);

  TensorDesc true_description = op.GetOutputDescByName("true_expected_count");
  true_description.SetShape(Shape(true_dimensions));
  true_description.SetDataType(DT_FLOAT);

  TensorDesc sampled_description = op.GetOutputDescByName("sampled_expected_count");
  sampled_description.SetShape(Shape(sampled_dimensions));
  sampled_description.SetDataType(DT_FLOAT);

  if (op.UpdateOutputDesc("sampled_candidates", candidate_description) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("fail to update output[sampled_candidates] desc"));
    return GRAPH_FAILED;
  }

  if (op.UpdateOutputDesc("true_expected_count", true_description) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("fail to update output[true_expected_count] desc"));
    return GRAPH_FAILED;
  }

  if (op.UpdateOutputDesc("sampled_expected_count", sampled_description) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("fail to update output[sampled_expected_count] desc"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge