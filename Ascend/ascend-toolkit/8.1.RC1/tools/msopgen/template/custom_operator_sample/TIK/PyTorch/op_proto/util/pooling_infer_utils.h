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
#include "../runtime/runtime_util.h"
#include "op_util.h"

using namespace ge;
namespace gert {
const size_t X_IDX = 0;
const size_t OUT_IDX = 0;
const size_t DIM_SIZE4 = 4;
const size_t SUPPORTED_DIM_NUM = 4;

static map<ge::Format, std::string> format2str = {
    {ge::Format::FORMAT_NCHW, "NCHW"}, {ge::Format::FORMAT_NHWC, "NHWC"}, {ge::Format::FORMAT_HWCN, "HWCN"},
    {ge::Format::FORMAT_DHWNC, "DHWNC"}, {ge::Format::FORMAT_DHWCN, "DHWCN"}, {ge::Format::FORMAT_NDHWC, "NDHWC"},
    {ge::Format::FORMAT_NCDHW, "NCDHW"}};

static bool GetDimInFormat(const std::string& opName, const std::string& formatStr, const std::string& dimName,
                           int64_t& dimPosition)
{
    dimPosition = formatStr.find(dimName);
    if (dimPosition < 0) {
        CUBE_INNER_ERR_REPORT(opName.c_str(), "Position(%s) is invalid: %ld, which format is %s.",
                              dimName.c_str(), dimPosition, formatStr.c_str());
        return false;
    }
    return true;
}

template <class T>
bool CheckPoolingPadsPositive(const std::string& opName,
                              const std::string& paddingModeStr,
                              T& inputs)
{
    int64_t pad_needed_h = 0;
    int64_t pad_needed_w = 0;
    if (paddingModeStr == "SAME") {
        pad_needed_h = (inputs.outH - 1) * inputs.strideH + inputs.windowH - inputs.inH;
        pad_needed_w = (inputs.outW - 1) * inputs.strideW + inputs.windowW - inputs.inW;
        if (pad_needed_h < 0  or pad_needed_w < 0) {
            std::string err_msg = OtherErrMsg("pad_needed should be positive");
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName, err_msg);
            return false;
        }
    }
    return true;
}

template <class T>
graphStatus UpdateOutputShape(const std::string& paddingModeStr,
                              const std::string& autoPadModeStr,
                              const int64_t ceilMode,
                              const std::string& opName,
                              T& inputs)
{
    inputs.outN = inputs.inN;
    inputs.outC = inputs.inC;

    if (paddingModeStr == "SAME") {
        inputs.outH = (inputs.inH + inputs.strideH - 1) / inputs.strideH;
        inputs.outW = (inputs.inW + inputs.strideW - 1) / inputs.strideW;
    } else if (paddingModeStr == "VALID") {
        if (ceilMode == 1) {
            inputs.outH = (inputs.inH - inputs.windowH) / inputs.strideH + 1;
            inputs.outW = (inputs.inW - inputs.windowW) / inputs.strideW + 1;
        } else {
            inputs.outH = (inputs.inH - inputs.windowH + inputs.strideH - 1) / inputs.strideH + 1;
            inputs.outW = (inputs.inW - inputs.windowW + inputs.strideW - 1) / inputs.strideW + 1;
        }
    } else {
        if (ceilMode == 1) {
            inputs.outH = (inputs.inH - inputs.windowH + inputs.padtop + inputs.padbottom) / inputs.strideH + 1;
            inputs.outW = (inputs.inW - inputs.windowW + inputs.padleft + inputs.padright) / inputs.strideW + 1;
        } else {
            inputs.outH = (inputs.inH - inputs.windowH + inputs.padtop + inputs.padbottom + inputs.strideH - 1)
                / inputs.strideH + 1;
            inputs.outW = (inputs.inW - inputs.windowW + inputs.padleft + inputs.padright + inputs.strideW - 1)
                / inputs.strideW + 1;
        }
    }
    // inorder to figure out whether is the avgpoolv2
    if (autoPadModeStr == "NOTSET") {
        bool positive = CheckPoolingPadsPositive(opName, paddingModeStr, inputs);
        OP_LOGE_IF(!positive, GRAPH_FAILED, opName, "check pooing pads positive failed.");
    }
    return GRAPH_SUCCESS;
}

template <class T>
graphStatus GetPoolingSumXShape(const gert::CompileTimeTensorDesc* tensorDescIn,
                                const std::string& opName,
                                const InferShapeContext* context,
                                const char* dataFormatPtr,
                                T& inputs)
{
    // Get fmap Dim
    const Format xdataFormat = tensorDescIn->GetOriginFormat();
    std::string dataFormat = format2str.at(xdataFormat);
    if ((dataFormat != "NCHW" && dataFormat != "NHWC") && (dataFormat != dataFormatPtr)) {
        std::string errMsg = OtherErrMsg("attr data format is wrong.");
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName, errMsg);
        return GRAPH_FAILED;
    }

    if (dataFormat.length() != DIM_SIZE4) {
        std::string errMsg =
            GetAttrValueErrMsg("Input format dim", std::to_string(dataFormat.length()), ConcatString(DIM_SIZE4));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName, errMsg);
        return GRAPH_FAILED;
    }

    bool get_dim_in_format = GetDimInFormat(opName, dataFormat, "N", inputs.xnPosition) &&
                             GetDimInFormat(opName, dataFormat, "C", inputs.xcPosition) &&
                             GetDimInFormat(opName, dataFormat, "H", inputs.xhPosition) &&
                             GetDimInFormat(opName, dataFormat, "W", inputs.xwPosition);
    if (!get_dim_in_format) {
        return GRAPH_FAILED;
    }

    const gert::Shape* shapeIn = context->GetInputShape(X_IDX);
    OP_LOGE_IF(shapeIn == nullptr, GRAPH_FAILED, opName, "fmap is null.");
    OP_LOGE_IF(shapeIn->GetDimNum() != SUPPORTED_DIM_NUM, GRAPH_FAILED, opName,
        "Not support input xShape dimnum %lu.", shapeIn->GetDimNum());

    // Set x shape into structure
    inputs.inN = shapeIn->GetDim(inputs.xnPosition);
    inputs.inC = shapeIn->GetDim(inputs.xcPosition);
    inputs.inH = shapeIn->GetDim(inputs.xhPosition);
    inputs.inW = shapeIn->GetDim(inputs.xwPosition);

    return GRAPH_SUCCESS;
}

template <class T>
graphStatus SetPoolingSumOutput(const std::string& opName, InferShapeContext* context, T& inputs)
{
    const gert::CompileTimeTensorDesc* tensordescOutput = context->GetOutputDesc(OUT_IDX);
    OP_LOGE_IF(tensordescOutput == nullptr, GRAPH_FAILED, opName, "Get output failed.");
    auto formatOut = tensordescOutput->GetOriginFormat();
    auto shapeOut = context->GetOutputShape(OUT_IDX);
    shapeOut->SetDimNum(SUPPORTED_DIM_NUM);
    // NC1HWC0(NCHW/NHWC)
    if (formatOut == Format::FORMAT_NCHW || formatOut == Format::FORMAT_NHWC) {
        shapeOut->SetDim(inputs.xnPosition, inputs.outN);
        shapeOut->SetDim(inputs.xcPosition, inputs.outC);
        shapeOut->SetDim(inputs.xhPosition, inputs.outH);
        shapeOut->SetDim(inputs.xwPosition, inputs.outW);
    } else {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName, "output y format is not correct! format should be NCHW or NHWC.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
}  // namespace gert
