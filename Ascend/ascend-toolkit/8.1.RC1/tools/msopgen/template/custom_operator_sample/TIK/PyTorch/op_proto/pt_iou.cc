/*
 * Copyright (C)  2020. Huawei Technologies Co., Ltd. All rights reserved.
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
 * \file pt_iou.cc
 * \brief
 */
#include "pt_iou.h"
#include <cmath>
#include <string>
#include <vector>


namespace ge {
// ----------------PTIou-------------------
IMPLEMT_COMMON_INFERFUNC(IouInferShape) {
    auto shape_box = op.GetInputDesc("bboxes").GetShape();
    auto shap_gbox = op.GetInputDesc("gtboxes").GetShape();
    vector<int64_t> shape_out;
    shape_out.push_back(shap_gbox.GetDim(0));
    shape_out.push_back(shape_box.GetDim(0));

    Shape output_shape(shape_out);
    DataType input_type = op.GetInputDesc("bboxes").GetDataType();

    TensorDesc td = op.GetOutputDesc("overlap");
    td.SetShape(output_shape);
    td.SetDataType(input_type);
    (void)op.UpdateOutputDesc("overlap", td);

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(PtIou, IouInferShape);

}  // namespace ge
