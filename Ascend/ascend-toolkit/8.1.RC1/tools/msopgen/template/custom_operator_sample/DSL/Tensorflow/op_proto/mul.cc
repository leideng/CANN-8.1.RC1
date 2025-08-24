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
 * \file mul.cc
 * \brief
 */
#include "mul.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "error_util.h"
#include "graph/utils/node_utils.h"
#include "register/infer_data_slice_registry.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
bool BroadCastTwoShape(const Operator& op, const ge::Shape& shape_x, const ge::Shape& shape_y,
                       std::vector<int64_t>& dim_out) {
    std::vector<int64_t> dim_x = shape_x.GetDims();
    std::vector<int64_t> dim_y = shape_y.GetDims();
    // exchange them
    if (dim_x.size() < dim_y.size()) {
        std::vector<int64_t> dim_tmp = dim_x;
        dim_x = dim_y;
        dim_y = dim_tmp;
    }

    // expand smalll shape
    if (dim_x.size() != dim_y.size()) {
        int dec = dim_x.size() - dim_y.size();
        for (int i = 0; i < dec; i++) {
            dim_y.insert(dim_y.begin(), (int64_t)1);
        }
    }

    // set out dims
    for (size_t i = 0; i < dim_x.size(); i++) {
        if ((dim_x[i] != dim_y[i]) && (dim_x[i] != 1) && (dim_y[i] != 1)) {
            OP_LOGE(op.GetName().c_str(), "The %s's dimensions does not match the broadcast rule(%lu %lu).",
                    op.GetName().c_str(), dim_x[i], dim_y[i]);
            return false;
        }

        int64_t dim = dim_x[i] > dim_y[i] ? dim_x[i] : dim_y[i];
        dim_out.push_back(dim);
    }
    return true;
}

bool InferShapeForMaximumAndMinimum(Operator& op) {
    auto attr_grad_x = false;
    auto attr_grad_y = false;
    if (op.GetAttr("grad_x", attr_grad_x) == GRAPH_FAILED) {
        OP_LOGE(op.GetName().c_str(), "get attr grad_x failed");
    }
    if (op.GetAttr("grad_y", attr_grad_y) == GRAPH_FAILED) {
        OP_LOGE(op.GetName().c_str(), "get attr grad_y failed");
    }
    if (attr_grad_x == false && attr_grad_y == false) {
        OP_LOGE(op.GetName().c_str(), "the grad_x and grad_y is not support all false");
      return false;
    }
    if (attr_grad_x) {
        if(!OneInOneOutDynamicInfer(op, "x1", {"y1"})) {
            return false;
        }
    }
    if (attr_grad_y) {
        if(!OneInOneOutDynamicInfer(op, "x2", {"y2"})) {
            return false;
        }
    }

    return true;
}

IMPLEMT_COMMON_INFERFUNC(TwoInOneOutCommonInferShape) {
    bool is_dynamic_output = true;
    if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(OneInOneOutCommonInferShape) {
    if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

static void InferElewiseTwoInput(vector<vector<int64_t>>& in_data_slice, const vector<vector<int64_t>> out_data_slice,
                                 const vector<int64_t> in_dims, const vector<int64_t> out_dims) {
    if (in_dims.size() == out_dims.size()) {
        for (size_t i = 0; i < in_dims.size(); i++) {
            if (in_dims[i] == 1) {
                in_data_slice.push_back({0, 1});
            } else {
                in_data_slice.push_back(out_data_slice[i]);
            }
        }
    } else {
        for (size_t i = 0; i < in_dims.size(); i++) {
            if (in_dims[i] == 1) {
                in_data_slice.push_back({0, 1});
            } else {
                in_data_slice.push_back(out_data_slice[out_dims.size() - in_dims.size() + i]);
            }
        }
    }
}

static void InferElewiseTwoInputdif(vector<vector<int64_t>>& in_data_slice, const vector<vector<int64_t>> out_data_slice,
                                    const vector<int64_t> in_dims, const vector<int64_t> out_dims, const int64_t aixs) {
    if (in_dims.size() == out_dims.size()) {
        for (size_t i = 0; i < in_dims.size(); i++) {
            if (in_dims[i] == 1) {
                in_data_slice.push_back({0, 1});
            } else {
                in_data_slice.push_back(out_data_slice[i]);
            }
        }
    } else if (in_dims.size() == 1) {
        in_data_slice.push_back({out_data_slice[aixs][0] * 16, out_data_slice[aixs][1] * 16});
    }
}

IMPLEMT_COMMON_INFER_DATA_SLICE(ElewiseTwoInputInferDataSlice) {
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    if (!op_desc) {
        OP_LOGW(op.GetName().c_str(), "GetOpDescFromOperator failed.");
        return GRAPH_FAILED;
    }

    auto tensor_desc_in_x1 = op_desc->MutableInputDesc("x1");
    if (!tensor_desc_in_x1) {
        OP_LOGW(op.GetName().c_str(), "Get input desc x1 failed.");
        return GRAPH_FAILED;
    }
    auto x1_shape = tensor_desc_in_x1->MutableShape();
    auto x1_format = tensor_desc_in_x1->GetFormat();
    std::vector<int64_t> x1_dims = x1_shape.GetDims();

    auto tensor_desc_in_x2 = op_desc->MutableInputDesc("x2");
    if (!tensor_desc_in_x2) {
        OP_LOGW(op.GetName().c_str(), "Get input desc x2 failed.");
        return GRAPH_FAILED;
    }
    auto x2_shape = tensor_desc_in_x2->MutableShape();
    auto x2_format = tensor_desc_in_x2->GetFormat();
    std::vector<int64_t> x2_dims = x2_shape.GetDims();

    auto tensor_desc_out_y = op_desc->MutableOutputDesc("y");
    if (!tensor_desc_out_y) {
        OP_LOGW(op.GetName().c_str(), "Get input desc y failed.");
        return GRAPH_FAILED;
    }
    auto y_shape = tensor_desc_out_y->MutableShape();
    auto y_format = tensor_desc_out_y->GetFormat();
    std::vector<int64_t> y_dims = y_shape.GetDims();

    vector<vector<int64_t>> y_data_slice = {};
    vector<vector<int64_t>> x1_data_slice = {};
    vector<vector<int64_t>> x2_data_slice = {};
    if (!ge::AttrUtils::GetListListInt(tensor_desc_out_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
        OP_LOGW(op.GetName().c_str(), "no data slice, use default as {}");
        return GRAPH_FAILED;
    }

    if (x1_format == x2_format) {
        InferElewiseTwoInput(x1_data_slice, y_data_slice, x1_dims, y_dims);
        InferElewiseTwoInput(x2_data_slice, y_data_slice, x2_dims, y_dims);
    } else {
        if ((x1_format == FORMAT_NC1HWC0 && (x2_dims.size() == 0 || x2_dims.size() == 1)) ||
            ((x1_dims.size() == 0 || x1_dims.size() == 1) && x2_format == FORMAT_NC1HWC0)) {
            // 5HD+ND
            InferElewiseTwoInputdif(x1_data_slice, y_data_slice, x1_dims, y_dims, 1);
            InferElewiseTwoInputdif(x2_data_slice, y_data_slice, x2_dims, y_dims, 1);
        } else if ((x1_format == FORMAT_FRACTAL_NZ && (x2_dims.size() == 0 || x2_dims.size() == 1)) ||
                   ((x1_dims.size() == 0 || x1_dims.size() == 1) && x2_format == FORMAT_FRACTAL_NZ)) {
            // NZ+ND
            InferElewiseTwoInputdif(x1_data_slice, y_data_slice, x1_dims, y_dims, y_dims.size() - 3);
            InferElewiseTwoInputdif(x2_data_slice, y_data_slice, x2_dims, y_dims, y_dims.size() - 3);
        } else if ((x1_format == FORMAT_FRACTAL_Z && (x2_dims.size() == 0 || x2_dims.size() == 1)) ||
                   ((x1_dims.size() == 0 || x1_dims.size() == 1) && x2_format == FORMAT_FRACTAL_Z)) {
            // F_Z+ND
            InferElewiseTwoInputdif(x1_data_slice, y_data_slice, x1_dims, y_dims, 0);
            InferElewiseTwoInputdif(x2_data_slice, y_data_slice, x2_dims, y_dims, 0);
        } else {
            for (size_t i = 0; i < x1_dims.size(); i++) {
                x1_data_slice.push_back({});
            }
            for (size_t i = 0; i < x2_dims.size(); i++) {
                x2_data_slice.push_back({});
            }
        }
    }

    if (!ge::AttrUtils::SetListListInt(tensor_desc_in_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice)) {
        OP_LOGW(op.GetName().c_str(), "data slice set failed");
        return GRAPH_FAILED;
    }
    if (!ge::AttrUtils::SetListListInt(tensor_desc_in_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice)) {
        OP_LOGW(op.GetName().c_str(), "data slice set failed");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(MaximumGradInferShape) {
    if (InferShapeForMaximumAndMinimum(op)) {
        return GRAPH_SUCCESS;
    }

    return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(MaximumGrad, MaximumGradInferShape);

IMPLEMT_COMMON_INFERFUNC(MinimumGradInferShape) {
    if (InferShapeForMaximumAndMinimum(op)) {
        return GRAPH_SUCCESS;
    }

    return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(MinimumGrad, MinimumGradInferShape);

IMPLEMT_VERIFIER(Mul, MulVerify) {
    if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Mul, ElewiseTwoInputInferDataSlice);
COMMON_INFER_FUNC_REG(Mul, TwoInOneOutCommonInferShape);
VERIFY_FUNC_REG(Mul, MulVerify);

}  // namespace ge