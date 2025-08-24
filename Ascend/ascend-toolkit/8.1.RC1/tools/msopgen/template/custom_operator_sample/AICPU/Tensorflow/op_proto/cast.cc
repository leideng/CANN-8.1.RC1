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
 * \file cast.cc
 * \brief
 */
#include "cast.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "error_util.h"

namespace ge {
// ----------------Cast-------------------
IMPLEMT_COMMON_INFERFUNC(CastInferShape) {
  // get input desc
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  vector<int64_t> input_shape = input_desc->MutableShape().GetDims();

  auto output_desc = op_info->MutableOutputDesc("y");
  if (IsUnknown(input_shape)) {
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(input_shape, input_range);

    output_desc->SetShape(GeShape(input_shape));
    output_desc->SetOriginShape(GeShape(input_shape));
    output_desc->SetShapeRange(input_range);
  } else {
    output_desc->SetShape(GeShape(input_shape));
  }
  int type;
  if (op.GetAttr("dst_type", type) == GRAPH_SUCCESS) {
    output_desc->SetDataType((ge::DataType)type);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Cast, CastInferShape);
// --------------Cast END-----------------// --------------Cast END-----------------
}  // namespace ge
