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
 * \file less.cc
 * \brief
 */
#include "less.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "error_util.h"

namespace ge {
// --------------------Less--------------------
IMPLEMT_VERIFIER(Less, LessVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(LessInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }

  op_desc->MutableOutputDesc("y")->SetDataType(DT_BOOL);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Less, LessInferShape);
VERIFY_FUNC_REG(Less, LessVerify);
// -----------------Less END-----------------------
}  // namespace ge
