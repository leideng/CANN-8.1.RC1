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
#include <algorithm>
#include "lp_norm.h"

namespace ge {
// ----------------LpNorm Begin-------------------
IMPLEMT_VERIFIER(LpNorm, LpNormVerify) { return GRAPH_SUCCESS; }
IMPLEMT_COMMON_INFERFUNC(LpNormInfer) {
    auto tensor_input = op.GetInputDesc("x");
    Shape x_shape = tensor_input.GetShape();
    DataType x_type = tensor_input.GetDataType();
    Format x_format = tensor_input.GetFormat();
    size_t dim_num = op.GetInputDesc("x").GetShape().GetDimNum();
    std::vector<int64_t> x_axes = {};
    std::vector<int64_t> new_axes = {};
    std::vector<int64_t> y_vec = {};
    std::vector<int64_t> x_dim_members = x_shape.GetDims();
    bool keep_dim = false;
    int32_t indice;
    (void)op.GetAttr("keepdim", keep_dim);
    if (x_axes.empty()) {
        for (int32_t i = 0; i < dim_num; i++) {
            new_axes.push_back(i);
        }
    } else {
        for (int32_t i = 0; i < x_axes.size(); i++) {
            indice = (x_axes[i] < 0) ? (x_axes[i] + dim_num) : x_axes[i];
            new_axes.push_back(indice);
        }
    }
    for (int32_t i = 0; i < x_shape.GetDimNum(); i++) {
        if (find(new_axes.begin(), new_axes.end(), i) != new_axes.end()) {
            if (keep_dim == true) {
                y_vec.push_back(1);
            }
    } else {
        y_vec.push_back(x_dim_members[i]);
        }
    }

    ge::Shape output_shape(y_vec);
    // update output desc
    ge::TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetShape(output_shape);
    output_desc.SetDataType(x_type);
    output_desc.SetFormat(x_format);
    (void)op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(LpNorm, LpNormInfer);
VERIFY_FUNC_REG(LpNorm, LpNormVerify);
// ----------------LpNorm END---------------------
}  // namespace ge