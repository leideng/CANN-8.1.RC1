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
#include "fill_v2d.h"
#include "op_log.h"

namespace ge {
// ----------------FillV2D Begin-------------------
IMPLEMT_INFERFUNC(FillV2D, FillV2DInferShape) {
  const int DIM_SIZE1 = 1;
  const int DIM_SIZE8 = 8;
  std::vector<int64_t> vec_dim;
  if (ge::GRAPH_SUCCESS != op.GetAttr("dims", vec_dim)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr failed of FillD!");
    return GRAPH_FAILED;
  }

  OP_LOGI(op.GetName().c_str(), "start infershape");

  if (vec_dim.size() < DIM_SIZE1 || vec_dim.size() > DIM_SIZE8) {
    OP_LOGE(op.GetName().c_str(), "dims must between 1 and 8.");
    return GRAPH_FAILED;
  }

  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(Shape(vec_dim));
  td.SetDataType(DT_FLOAT);

  op.UpdateOutputDesc("y", td);
  OP_LOGI(op.GetName().c_str(), "infershape success!");
  return GRAPH_SUCCESS;
}

// Registered inferfunction
INFER_FUNC_REG(FillV2D, FillV2DInferShape);
// ----------------FillV2D END---------------------
}  // namespace ge