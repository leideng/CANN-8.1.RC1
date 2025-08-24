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
 * \file batch_norm.cc
 * \brief
 */
#include "batch_norm.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "error_util.h"
#include "graph/utils/node_utils.h"

namespace ge {
// -----------------------------BatchNorm------------------------------
IMPLEMT_VERIFIER(BatchNorm, BatchNormVerify) {
  if (!CheckTwoInputDtypeSame(op, "scale", "offset")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BatchNorm, BatchNormInferShape) {
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NHWC" && data_format != "NCHW") {
      string expected_format_list = ConcatString("NHWC, NCHW");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  if (!OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "scale", {"batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"})) {
    return GRAPH_FAILED;
  }
  std::vector<int64_t> oShapeVector;
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto output_desc = op_info->MutableOutputDesc("reserve_space_3");
  if (output_desc != nullptr) {
    output_desc->SetShape(GeShape(oShapeVector));
    output_desc->SetDataType(DT_FLOAT);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BatchNorm, BatchNormInferShape);
VERIFY_FUNC_REG(BatchNorm, BatchNormVerify);
// -----------------------------BatchNorm END----------------------------
}  // namespace ge