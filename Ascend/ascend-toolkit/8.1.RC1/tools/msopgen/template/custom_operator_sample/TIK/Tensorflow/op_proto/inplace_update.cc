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

#include "inplace_update.h"

namespace ge {
// ----------------InplaceUpdate-------------------
IMPLEMT_COMMON_INFERFUNC(InplaceUpdateInferShape) {
  auto output_desc = op.GetInputDesc("x");
  auto output_shape_dims = output_desc.GetShape().GetDims();
  Shape output_shape(output_shape_dims);
  output_desc.SetShape(output_shape);

  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InplaceUpdate, InplaceUpdateInferShape);
// ----------------InplaceUpdate END-------------------
}  // namespace ge