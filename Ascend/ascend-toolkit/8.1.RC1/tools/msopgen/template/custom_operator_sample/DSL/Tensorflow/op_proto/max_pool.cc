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
 * \file max_pool.cc
 * \brief
 */
/* reslove the complexity of pooling fuction. */
#include "max_pool.h"
#include <string.h>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include "graph/operator.h"
#include "op_log.h"
#include "common/util/error_manager/error_manager.h"
#include "util/common_shape_fns.h"
#include "error_util.h"
#include "util/util.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {

// ----------------MaxPool-------------------
static void UpdateDimAndRange(const int64_t& ksize, const int64_t& strides, int64_t& dim_size,
                              std::pair<int64_t, int64_t>& dim_range) {
    if (dim_size != -1) {
        int64_t output_dim_size = (dim_size - ksize + strides) / strides;
        dim_range = std::pair<int64_t, int64_t>{output_dim_size, output_dim_size};
        dim_size = output_dim_size;
    } else {
        int64_t first_range = dim_range.first == 1 ? 1 : (dim_range.first - ksize + strides) / strides;
        int64_t second_range = dim_range.second == -1 ? -1 : (dim_range.second - ksize + strides) / strides;
        dim_range = std::pair<int64_t, int64_t>{first_range, second_range};
    }
}

IMPLEMT_INFERFUNC(MaxPool, MaxPoolInferShape) {
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    auto input_desc = op_info->MutableInputDesc("x");
    auto input_shape = input_desc->MutableShape();
    auto input_format = input_desc->GetFormat();
    auto input_dtype = input_desc->GetDataType();
    auto output_desc = op_info->MutableOutputDesc("y");
    output_desc->SetDataType(input_dtype);
    // get input ksize
    std::vector<int32_t> ksize;
    if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed!");
        return GRAPH_FAILED;
    }
    // get input strides
    std::vector<int32_t> strides;
    if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed!");
        return GRAPH_FAILED;
    }
    // get input data_format
    std::string data_format;
    if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr data_format failed!");
        return GRAPH_FAILED;
    }
    // get input padding
    std::string padding;
    if (GRAPH_SUCCESS != op.GetAttr("padding", padding)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> input_dims = input_shape.GetDims();
    std::vector<std::pair<int64_t, int64_t>> input_range;

    // dynamic case, input shape is -2, output is [-1, -1, -1, -1], only support NHWC or NCHW
    if (IsUnknownRankShape(input_dims)) {
        OP_LOGW(op.GetName().c_str(), "the input os unkown rank, will set the input [-1, -1, -1, -1].");
        input_dims = {-1, -1, -1, -1};
    } else {
        input_desc->GetShapeRange(input_range);
    }
    MakeUpShapeRange(input_dims, input_range);

    // set output shape
    std::vector<int64_t> output_dims;
    std::vector<std::pair<int64_t, int64_t>> output_range;

    auto input_h_dim = input_format == FORMAT_NHWC ? 1 : 2;
    auto input_w_dim = input_format == FORMAT_NHWC ? 2 : 3;
    auto strides_h_dim = data_format == "NHWC" ? 1 : 2;
    auto strides_w_dim = data_format == "NHWC" ? 2 : 3;

    if (padding != "VALID") {
        ksize[strides_h_dim] = 1;
        ksize[strides_w_dim] = 1;
    }

    for (size_t i = 0; i < input_dims.size(); i++) {
        int64_t dim_size = input_dims[i];
        auto dim_range = input_range[i];
        if (i == input_h_dim) {
            UpdateDimAndRange(ksize[strides_h_dim], strides[strides_h_dim], dim_size, dim_range);
        } else if (i == input_w_dim) {
            UpdateDimAndRange(ksize[strides_w_dim], strides[strides_w_dim], dim_size, dim_range);
        }
        output_dims.push_back(dim_size);
        output_range.push_back(dim_range);
    }

    output_desc->SetShape(GeShape(output_dims));
    output_desc->SetShapeRange(output_range);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MaxPool, MaxPoolVerify) {
    // check ksize
    std::vector<int32_t> ksize;
    if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed!");
        return GRAPH_FAILED;
    }
    if (ksize.size() != 4) {
        OP_LOGE(op.GetName().c_str(), "The length of ksize must be equal to the length of shape!");
        return GRAPH_FAILED;
    }
    // check strides
    std::vector<int32_t> strides;
    if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed!");
        return GRAPH_FAILED;
    }
    if (strides.size() != 4) {
        OP_LOGE(op.GetName().c_str(), "The length of strides must be equal to the length of shape!");
        return GRAPH_FAILED;
    }
    // check data_format
    std::string data_format;
    if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr data_format failed!");
        return GRAPH_FAILED;
    }
    if (data_format != "NHWC" && data_format != "NCHW" && data_format != "NC1HWC0") {
        string expected_format_list = ConcatString("NHWC,NCHW,NC1HWC0");
        OP_LOGE(op.GetName().c_str(), "data_format only support 'NHWC','NCHW' and 'NC1HWC0'.");
        return GRAPH_FAILED;
    }
    if (data_format == "NHWC") {
        if ((ksize[0] != 1) || (ksize[3] != 1) || (strides[0] != 1) || (strides[3] != 1)) {
            OP_LOGE(op.GetName().c_str(), "Pooling across width/height and other ksize dimension should be one");
            return GRAPH_FAILED;
        }
    }
    if ((data_format == "NCHW") || (data_format == "NC1HWC0")) {
        if ((ksize[0] != 1) || (ksize[1] != 1) || (strides[0] != 1) || (strides[1] != 1)) {
            OP_LOGE(op.GetName().c_str(), "Pooling across width/height and other ksize dimension should be one");
            return GRAPH_FAILED;
        }
    }
    // check padding
    std::string padding;
    if (GRAPH_SUCCESS != op.GetAttr("padding", padding)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
        return GRAPH_FAILED;
    }
    if (padding != "SAME" && padding != "VALID") {
        string expected_format_list = ConcatString("SAME,VALID");
        OP_LOGE(op.GetName().c_str(), "padding only support SAME or VALID padding mode!");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

static void InferHWMaxPool(int64_t kernel, int64_t stride, vector<int64_t>& output, vector<int64_t>& input,
                           int64_t& ori_input) {
    int64_t first_start = output[0] * stride;
    int64_t second_start = output[1] * stride;
    int64_t first_end = std::min(first_start + kernel, ori_input);
    int64_t second_end = std::min(second_start + kernel, ori_input);
    int64_t start = std::max(first_start, int64_t(0));
    int64_t end = second_end - 1;
    input = {start, end};
}

IMPLEMT_INFER_DATA_SLICE(MaxPool, MaxPoolInferDataSlice) {
    auto inputTensorDesc = op.GetInputDesc("x");
    auto shape = inputTensorDesc.GetShape();
    std::vector<int64_t> dims_input = shape.GetDims();

    std::vector<int64_t> ksizeList;
    std::vector<int64_t> stridesList;
    std::string dataFormat;
    std::string paddingMode;
    op.GetAttr("ksize", ksizeList);
    op.GetAttr("strides", stridesList);
    op.GetAttr("data_format", dataFormat);
    op.GetAttr("padding", paddingMode);

    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t windowH = 0;
    int64_t windowW = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t dilationH = 0;

    if (dataFormat == "NHWC") {
        inputH = dims_input[1];
        inputW = dims_input[2];
        windowH = ksizeList[1];
        windowW = ksizeList[2];
        strideH = stridesList[1];
        strideW = stridesList[2];
    } else if (dataFormat == "NCHW") {
        inputH = dims_input[2];
        inputW = dims_input[3];
        windowH = ksizeList[2];
        windowW = ksizeList[3];
        strideH = stridesList[2];
        strideW = stridesList[3];
    }

    if (dataFormat == "NHWC" && ksizeList[0] == inputH && ksizeList[1] == inputW) {
        return NO_OVERLAP_DIM;
    }
    if (dataFormat == "NCHW" && ksizeList[0] == inputH && ksizeList[1] == inputW) {
        return NO_OVERLAP_DIM;
    }
    if (paddingMode == "SAME") {
        return NO_OVERLAP_DIM;
    }

    vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}};
    vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    GeTensorDescPtr tensor_desc_out = op_desc->MutableOutputDesc("y");
    GeTensorDescPtr tensor_desc_in = op_desc->MutableInputDesc("x");
    if (!ge::AttrUtils::GetListListInt(tensor_desc_out, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
        OP_LOGI(op.GetName().c_str(), "no data slice, use default as {{}, {}, {}, {}, {}}");
        return GRAPH_FAILED;
    }

    for (unsigned i = 0; i < y_data_slice.size(); i++) {
        if (y_data_slice[i].size() > 0) {
            if (i == 0) {
                return NO_OVERLAP_DIM;
            } else if (i == 1 or i == 3 or i == 4) {
                return NOT_SUPPORT_SLICE;
            } else if (i == 2) {
                vector<int64_t> input_h;
                InferHWMaxPool(windowH, strideH, y_data_slice[i], input_h, inputH);
                x_data_slice[i] = input_h;
            }
        }
    }

    for (unsigned i = 0; i < x_data_slice.size(); i++) {
        if (x_data_slice[i].size() > 0) {
            if (!AttrUtils::SetListListInt(tensor_desc_in, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
                return GRAPH_FAILED;
            }
            return GRAPH_SUCCESS;
        }
        return NO_OVERLAP_DIM;
    }

    return NO_OVERLAP_DIM;
}

INFER_FUNC_REG(MaxPool, MaxPoolInferShape);
VERIFY_FUNC_REG(MaxPool, MaxPoolVerify);
INFER_DATA_SLICE_FUNC_REG(MaxPool, MaxPoolInferDataSlice);
}  // namespace ge