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
 * \file conv2d.cpp
 * \brief
 */
 
 #define CHECK_FORMAT(format)                                                     \
  {                                                                              \
    if (ge::FORMAT_RESERVED == format) {                                      \
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "get format failed:%s:%d", #format, format); \
      return false;                                                              \
    }                                                                            \
  }

#define CHECK_FORMAT_V2(format)                                                  \
  {                                                                              \
    if (ge::FORMAT_RESERVED == format) {                                      \
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "get format failed:%s:%d", #format, format); \
      return GRAPH_FAILED;                                                       \
    }                                                                            \
  }

#include "conv2d.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "error_util.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "util/util.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "axis_util.h"

namespace ge {

namespace {
  const int32_t kConv3dDimSizeLimit = 5;
  const int32_t kConv3dLengthPadsLimit = 6;
  const int32_t kConv3dStridesSizeLimit = 5;
  const int32_t kConv3dInputSizeLimit = 5;
  const int32_t kConv3dPadsSizeLimit = 6;
  const int32_t kConv3dDataSlice = 6;
  const int32_t kDeformDimSizeLimit = 4;
  const int32_t kDeformKsizeLimit = 2;
  const int64_t kDynamicRangeLowerBound = 1;
  const int64_t kDynamicRangeUpperBound = 4096;
  const char* const kPreOpInputShapeRange = "_pre_op_in_range";
  const char* const kForceInfershapeWhenRunning = "_force_infershape_when_running";
}

// --------------------------Conv2D------------------------------
/*!
  * @brief Convert different framework pad param to ir pads:
  *
  * [_padding]: 4D lsit, format sensitive, need convert to pads
  * [padding]: 'SAME' or 'VALID', need convert to pads
  * [pads]: 4D list, format sensitive, no need convert
  *
  * @param op Conv2D operator.
  * @param ih, iw  Input images H/W size.
  * @param kh, kw  Input filter H/W size.
  * @param strh, strw  Input stride H/W value.
  * @param dilh, dilw  Input dilation H/W value.
  * @param padt, padb, padl, padr Top, bottom, left, right padding.
  * @return bool Whether the pads setting is correct.
  */
static bool GetPadConv2D(ge::Operator& op, int32_t ih, int32_t iw, int32_t kh, int32_t kw, int32_t strh, int32_t strw,
                         int32_t dilh, int32_t dilw, int32_t& padt, int32_t& padb, int32_t& padl, int32_t& padr) {
  std::string pad_str;
  std::vector<int32_t> pad_list;
  if (GRAPH_SUCCESS == op.GetAttr("padding", pad_str) && pad_str.compare("EXPLICIT") != 0) {
    if (pad_str.compare("SAME") == 0) {
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t dkh = dilh * (kh - 1) + 1;
      int32_t dkw = dilw * (kw - 1) + 1;
      int32_t pad_h = std::max((tails_h > 0 ? dkh - tails_h : dkh - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? dkw - tails_w : dkw - strw), 0);
      pad_list.push_back(pad_h / 2);
      pad_list.push_back(pad_h / 2 + pad_h % 2);
      pad_list.push_back(pad_w / 2);
      pad_list.push_back(pad_w / 2 + pad_w % 2);
    } else if (pad_str.compare("VALID") == 0) {
      pad_list.push_back(0);
      pad_list.push_back(0);
      pad_list.push_back(0);
      pad_list.push_back(0);
    } else {
      OP_LOGE(op.GetName().c_str(),
              "padding should be SAME or VALID."
              " actual is: %s.",
              pad_str.c_str());
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["expected_pad_mode"] = "SAME or VALID";
      err_map["actual_pad_mode"] = pad_str;
      std::string report_error_code = "E50050";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    OP_LOGD(op.GetName().c_str(),
            "pads info is [%d,%d,%d,%d].",
            pad_list[0], pad_list[1], pad_list[2], pad_list[3]);
    op.SetAttr("pads", pad_list);
  }

  // handle attr auto_pad from ONNX
  if (GRAPH_SUCCESS == op.GetAttr("auto_pad", pad_str)) {
    if (pad_str.compare("SAME_UPPER") == 0) {
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t dkh = dilh * (kh - 1) + 1;
      int32_t dkw = dilw * (kw - 1) + 1;
      int32_t pad_h = std::max((tails_h > 0 ? dkh - tails_h : dkh - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? dkw - tails_w : dkw - strw), 0);
      pad_list.push_back(pad_h / 2);
      pad_list.push_back(pad_h / 2 + pad_h % 2);
      pad_list.push_back(pad_w / 2);
      pad_list.push_back(pad_w / 2 + pad_w % 2);
      op.SetAttr("pads", pad_list);
    } else if (pad_str.compare("SAME_LOWER") == 0) {
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t dkh = dilh * (kh - 1) + 1;
      int32_t dkw = dilw * (kw - 1) + 1;
      int32_t pad_h = std::max((tails_h > 0 ? dkh - tails_h : dkh - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? dkw - tails_w : dkw - strw), 0);
      pad_list.push_back(pad_h / 2 + pad_h % 2);
      pad_list.push_back(pad_h / 2);
      pad_list.push_back(pad_w / 2 + pad_w % 2);
      pad_list.push_back(pad_w / 2);
      op.SetAttr("pads", pad_list);
    } else if (pad_str.compare("NOTSET") == 0) {
    } else if (pad_str.compare("VALID") == 0) {
      pad_list.push_back(0);
      pad_list.push_back(0);
      pad_list.push_back(0);
      pad_list.push_back(0);
      op.SetAttr("pads", pad_list);
    } else {
      OP_LOGE(op.GetName().c_str(),
              "padding should be SAME or VALID."
              " actual is: %s.",
              pad_str.c_str());
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["expected_pad_mode"] = "NOTSET, SAME_UPPER, SAME_LOWER or VALID";
      err_map["actual_pad_mode"] = pad_str;
      std::string report_error_code = "E50050";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);

      return false;
    }
  }

  std::vector<int32_t> pads_list;
  op.GetAttr("pads", pads_list);
  auto p_size = pads_list.size();
  if (pads_list.empty() || p_size != 4) {
    OP_LOGE(op.GetName().c_str(), "pads list should be 4D. actual is: %u.", p_size);
    map<string, string> err_map;
    err_map["param_name"] = "pads";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(p_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  padt = pads_list[0];
  padb = pads_list[1];
  padl = pads_list[2];
  padr = pads_list[3];
  if(op.GetOpType() == "Conv2D") {
    int32_t ho = (ih + padt + padb - (kh - 1) * dilh - 1) / strh + 1;
    int32_t wo = (iw + padl + padr - (kw - 1) * dilw - 1) / strw + 1;
    int32_t hr = (ih + padt + padb - (kh - 1) * dilh - 1) % strh;
    int32_t wr = (iw + padl + padr - (kw - 1) * dilw - 1) % strw;
    if ((ho == 1 && hr <= padb) || (wo == 1 && wr <= padr)) {
      if (ho == 1 && hr <= padb) {
          padb -= hr;
          pads_list[1] = padb;
      }
      if (wo == 1 && wr <= padr) {
          padr -= wr;
          pads_list[3] = padr;
      }
      op.SetAttr("pads", pads_list);
    }
  }
  auto x_shape = op.GetInputDesc("x").GetShape().GetDims();
  bool negative_pad = (padt < 0 || padb < 0 || padl < 0 || padr < 0);
  bool unknown_rank = IsUnknownRankShape(x_shape);
  bool unknown_shape = IsUnKnownShape(x_shape);

  if ((!unknown_shape) && (!unknown_rank) && negative_pad) {
    OP_LOGE(op.GetName().c_str(),
            "pads should be positive, "
            " actual is [%d,%d,%d,%d].",
            padt, padb, padl, padr);
    map<string, string> err_map;
    err_map["param_name"] = "pads";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "positive";
    err_map["input_value"] =
        std::to_string(padt) + ", " + std::to_string(padb) + ", " + std::to_string(padl) + ", " + std::to_string(padr);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

/*!
  * @brief Get 2D(H/W) stride and dilation params to infershape output.
  *
  * [strides]: 4D list, format sensitive, according to first input tensor format
  * [dilations]: 4D list, format sensitive
  *
  * @param op Conv2D operator.
  * @param refer Valid value reference format.
  * @param strh, strw  Input stride H/W value.
  * @param dilh, dilw  Input dilation H/W value.
  * @return bool Whether the strides, dilations settings are correct.
  */
static bool GetAttrsConv2D(ge::Operator& op, Format refer, int32_t& strh, int32_t& strw, int32_t& dilh, int32_t& dilw) {
  std::vector<int32_t> stride_list;
  op.GetAttr("strides", stride_list);
  auto s_size = stride_list.size();
  if (stride_list.empty() || s_size != 4) {
    OP_LOGE(op.GetName().c_str(), "strides list should be 4D. actual is: %u.", s_size);
    map<string, string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(s_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  std::vector<int32_t> dilation_list;
  op.GetAttr("dilations", dilation_list);
  auto d_size = dilation_list.size();
  if (dilation_list.empty() || d_size != 4) {
    OP_LOGE(op.GetName().c_str(), "dilations list should be 4D. actual is: %u.", d_size);
    map<string, string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(d_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (refer == FORMAT_NCHW) {
    strh = stride_list[2];
    strw = stride_list[3];
    dilh = dilation_list[2];
    dilw = dilation_list[3];
  } else if (refer == FORMAT_NHWC) {
    strh = stride_list[1];
    strw = stride_list[2];
    dilh = dilation_list[1];
    dilw = dilation_list[2];
  }
  if (strh <= 0 || strw <= 0) {
    OP_LOGE(op.GetName().c_str(),
            "strides should be positive,"
            " actual is [%d,%d].",
            strh, strw);
    map<string, string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "positive";
    err_map["input_value"] = std::to_string(strh) + ", " + std::to_string(strw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (dilh <= 0 || dilw <= 0) {
    OP_LOGE(op.GetName().c_str(),
            "dilations should be positive,"
            " actual is [%d,%d].",
            dilh, dilw);
    map<string, string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "positive";
    err_map["input_value"] = std::to_string(dilh) + ", " + std::to_string(dilw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

static void SetConv2dOutShapeRange(const std::string& pad_str,
                                   size_t idx,
                                   const vector<int32_t>& attr_params,
                                   const std::vector<std::pair<int64_t, int64_t>>& fm_range,
                                   std::vector<std::pair<int64_t, int64_t>>& out_range) {
  size_t attr_idx = 0;
  int32_t stride = attr_params[attr_idx++];
  int32_t dilation = attr_params[attr_idx++];
  int32_t pad = attr_params[attr_idx++];
  int32_t kernel = attr_params[attr_idx++];
  int32_t low = fm_range[idx].first;
  int32_t high = fm_range[idx].second;
  if (pad_str == "SAME") {
    out_range[idx].first = (low + stride -1) / stride;
    out_range[idx].second = (high + stride -1) / stride;
  } else {
    out_range[idx].first = (low + pad - dilation * (kernel - 1) - 1) / stride + 1;
    out_range[idx].second = (high + pad - dilation * (kernel - 1) - 1) / stride + 1;
  }
  out_range[idx].first = std::max(out_range[idx].first, kDynamicRangeLowerBound);
  out_range[idx].second = std::min(out_range[idx].second, kDynamicRangeUpperBound);
  if(high == -1) {
    out_range[idx].second = high;
  }
}

static bool SetConv2dOutShapeRange(op::Conv2D& op,
                                   const vector<int32_t>& attr_params,
                                   vector<int64_t>& y_shape,
                                   ge::GeTensorDescPtr& x_tensor,
                                   ge::GeTensorDescPtr& y_tensor) {
  auto x_shape = x_tensor->MutableShape().GetDims();
  auto x_format = x_tensor->GetFormat();

  size_t idx = 0;
  int32_t strh = attr_params[idx++];
  int32_t strw = attr_params[idx++];
  int32_t dilh = attr_params[idx++];
  int32_t dilw = attr_params[idx++];
  int32_t padt = attr_params[idx++];
  int32_t padb = attr_params[idx++];
  int32_t padl = attr_params[idx++];
  int32_t padr = attr_params[idx++];
  int32_t kn = attr_params[idx++];
  int32_t kh = attr_params[idx++];
  int32_t kw = attr_params[idx++];

  size_t idx_n = 0;
  size_t idx_h = 0;
  size_t idx_w = 0;
  size_t idx_c = 0;
  if (x_format == FORMAT_NHWC) {
    idx_h = 1;
    idx_w = 2;
    idx_c = 3;
  } else if (x_format == FORMAT_NCHW) {
    idx_c = 1;
    idx_h = 2;
    idx_w = 3;
  }

  // update pads if padding is SAME
  std::string pad_str;
  if (!x_shape.empty() && GRAPH_SUCCESS == op.GetAttr("padding", pad_str) && pad_str == "SAME" &&
      (x_shape[idx_h] == -1 or x_shape[idx_w] == -1)) {
    op.SetAttr("pads", {-1, -1, -1, -1});
    OP_LOGD(op.GetName().c_str(), "set pads to {-1, -1, -1, -1} when padding is SAME in dynamic_shape");
  }

  OP_LOGD(op.GetName().c_str(), "dynamic shape set range");
  std::vector<std::pair<int64_t, int64_t>> fm_range;
  x_tensor->GetShapeRange(fm_range);
  if (x_shape[idx_h] == -1) {
    y_shape[idx_h] = -1;
  }
  if (x_shape[idx_w] == -1) {
    y_shape[idx_w] = -1;
  }
  if (!fm_range.empty()) {
    for (size_t i = 0; i < fm_range.size(); i++) {
      OP_LOGD(op.GetName().c_str(), "fmap Range[%u] is (%lld, %lld)", i, fm_range[i].first, fm_range[i].second);
    }

    std::vector<std::pair<int64_t, int64_t>> out_range(fm_range);
    out_range[idx_c] = std::make_pair((int64_t)kn, (int64_t)kn);
    out_range[idx_h] = std::make_pair((int64_t)y_shape[idx_h], (int64_t)y_shape[idx_h]);
    out_range[idx_w] = std::make_pair((int64_t)y_shape[idx_w], (int64_t)y_shape[idx_w]);
    if (x_shape[idx_h] == -1) {
      vector<int32_t> attr_params_h = {strh, dilh, padt + padb, kh};
      SetConv2dOutShapeRange(pad_str, idx_h, attr_params_h, fm_range, out_range);
    }
    if (x_shape[idx_w] == -1) {
      vector<int32_t> attr_params_w = {strw, dilw, padl + padr, kw};
      SetConv2dOutShapeRange(pad_str, idx_w, attr_params_w, fm_range, out_range);
    }
    y_tensor->SetShapeRange(out_range);
    for (size_t i = 0; i < out_range.size(); i++) {
      OP_LOGD(op.GetName().c_str(), "output Range[%u] is (%lld, %lld)", i, out_range[i].first, out_range[i].second);
    }
  }
  y_tensor->SetShape(GeShape(y_shape));
  return true;
}

/*!
  * Simply get value range in grade list.
  */
static bool GetSingleRange(ge::Operator& op, const std::vector<int64_t>& grade,
                          const int64_t& value, int64_t& low, int64_t& high) {
  size_t min_size = 2;
  if (grade.size() < min_size) {
    OP_LOGE(op.GetName().c_str(), "input grade size smaller then %u", min_size);
    return false;
  }
  // grade is in ascending order
  size_t last = grade.size() - 1;
  if (value > grade[last]) {
    OP_LOGE(op.GetName().c_str(), "input value %lld is out the range of %lld", value, grade[last]);
    return false;
  }
  // if it is the right boundary value, use the right closed interval
  if (value == grade[last]) {
    low = grade[last - 1];
    high = grade[last];
    return true;
  }
  for (auto n : grade) {
    if (value >= n) {
      low = n;
    }
    if (value < n) {
      high = n;
      break;
    }
  }
  return true;
}

/*!
  * Generate NHW shape range
  */
static bool GenConv2dShapeRange(ge::Operator& op, ge::GeTensorDescPtr& x_tensor,
                                std::vector<std::pair<int64_t, int64_t>>& input_range) {
  auto x_shape = x_tensor->MutableShape().GetDims();
  // only support 4D shape
  auto x_format = x_tensor->GetFormat();
  size_t idx_n = 0;
  size_t idx_h = 0;
  size_t idx_w = 0;
  size_t idx_c = 0;
  if (x_format == FORMAT_NHWC) {
    idx_h = 1;
    idx_w = 2;
    idx_c = 3;
  } else {
    idx_c = 1;
    idx_h = 2;
    idx_w = 3;
  }
  std::vector<int64_t> grade_n = {1, 2, 4, 8, 16, 32, ((1 << 31) - 1)};
  std::vector<int64_t> grade_h = {1, 4, 16, 32, 64, 128, 192, 256, 512, 768, 1024, 4096};
  std::vector<int64_t> grade_w = {1, 4, 16, 32, 64, 128, 192, 256, 512, 768, 1024, 4096};
  // init empty range
  // shape -1 without set range call "GetShapeRange" will return [1,-1]
  input_range = {{}, {}, {}, {}};
  std::vector<std::pair<int64_t, int64_t>> range_set;
  x_tensor->GetShapeRange(range_set);
  std::map<size_t, std::vector<int64_t>> grade_map;
  grade_map[idx_n] = grade_n;
  grade_map[idx_h] = grade_h;
  grade_map[idx_w] = grade_w;
  for (auto item: grade_map) {
    // allow shape -1 with range
    if(x_shape[item.first] == -1) {
      if (range_set.size() > item.first) {
        input_range[item.first] = range_set[item.first];
      } else {
        OP_LOGE(op.GetName().c_str(), "cant't get input index %zu range", item.first);
        return false;
      }
    } else {
      int64_t low = 1;
      int64_t high = 1;
      if (!GetSingleRange(op, item.second, x_shape[item.first], low, high)) {
        OP_LOGE(op.GetName().c_str(), "failed to get the %zu range", item.first);
        return false;
      }
      input_range[item.first] = std::make_pair(low, high);
    }
  }
  input_range[idx_c] = (std::make_pair(x_shape[idx_c], x_shape[idx_c]));
  return true;
}

/*!
  * @brief Infer output shape and dtype, dtype is same to first input tensor, Output
  *        format is set by ge parser process already.
  * @param Conv2DInfer Conv2D infershape function.
  * @return Status The processing flow result.
  */
IMPLEMT_INFERFUNC(Conv2D, Conv2DInfer) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv2DInfer.");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_tensor = op_desc->MutableInputDesc("x");
  auto w_tensor = op_desc->MutableInputDesc("filter");

  auto x_shape = x_tensor->MutableShape().GetDims();
  auto w_shape = w_tensor->MutableShape().GetDims();
  bool unknown_rank = IsUnknownRankShape(x_shape);
  if ((!unknown_rank && x_shape.size() != 4) || w_shape.size() != 4) {
    return GRAPH_FAILED;
  }
  auto x_format = x_tensor->GetFormat();
  auto w_format = w_tensor->GetFormat();
  CHECK_FORMAT(x_format);
  CHECK_FORMAT(w_format);

  int32_t in = -1;
  int32_t ic = -1;
  int32_t ih = -1;
  int32_t iw = -1;
  int32_t kn = 0;
  int32_t kc = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  if (x_format == FORMAT_NCHW) {
    if (!unknown_rank) {
      in = x_shape[0];
      ic = x_shape[1];
      ih = x_shape[2];
      iw = x_shape[3];
    }
  } else if (x_format == FORMAT_NHWC) {
    if (!unknown_rank) {
      in = x_shape[0];
      ic = x_shape[3];
      ih = x_shape[1];
      iw = x_shape[2];
    }
  } else {
    OP_LOGE(op.GetName().c_str(),
            "input x format should be NCHW or NHWC."
            " actual is: %s",
            TypeUtils::FormatToSerialString(x_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "x";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_format_list"] = "NCHW or NHWC";
    err_map["format"] = TypeUtils::FormatToSerialString(x_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (w_format == FORMAT_NCHW) {
    kn = w_shape[0];
    kc = w_shape[1];
    kh = w_shape[2];
    kw = w_shape[3];
  } else if (w_format == FORMAT_NHWC) {
    kn = w_shape[0];
    kc = w_shape[3];
    kh = w_shape[1];
    kw = w_shape[2];
  } else if (w_format == FORMAT_HWCN) {
    kn = w_shape[3];
    kc = w_shape[2];
    kh = w_shape[0];
    kw = w_shape[1];
  } else {
    OP_LOGE(op.GetName().c_str(),
            "input filter format should be NCHW, NHWC or HWCN."
            " actual is: %s",
            TypeUtils::FormatToSerialString(w_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "filter";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_format_list"] = "NCHW, NHWC or HWCN";
    err_map["format"] = TypeUtils::FormatToSerialString(w_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  auto bias_tensor = op_desc->MutableInputDesc("bias");
  if(bias_tensor != nullptr) {
    auto bias_shape = bias_tensor->MutableShape().GetDims();
    if (bias_shape.size() == 1 && bias_shape[0] != kn) {
      OP_LOGE(op.GetName().c_str(), "input bias size should be equal to out_channels.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input bias size should be equal to out_channels.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    } else if (bias_shape.size() > 1) {
      OP_LOGE(op.GetName().c_str(), "input bias shape should be 1D.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input bias shape should be 1D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  }

  // set data_format: copy value of x_format to data_format
  // data_format will be used to infer position of H/W
  // in strides and dilations(previously used ori_format)
  std::string data_format;
  std::string attr_data_format = "data_format";
  std::string data_format_NCHW = "NCHW";
  std::string data_format_NHWC = "NHWC";

  if (GRAPH_SUCCESS == op.GetAttr(attr_data_format, data_format)) {
    OP_LOGI(data_format.c_str(), "conv before set data_format");
  }

  if (x_format == ge::FORMAT_NCHW) {
    op.SetAttr(attr_data_format, data_format_NCHW);
  } else {
    op.SetAttr(attr_data_format, data_format_NHWC);
  }

  op.GetAttr(attr_data_format, data_format);
  OP_LOGI(data_format.c_str(), "conv after set data_format");

  int64_t groups = 1;
  op.GetAttr("groups", groups);
  bool is_dynamic = false;
  // when static op or dynamic op phase_running, is_dynamic == False
  if (std::find(x_shape.begin(), x_shape.end(), -1) != x_shape.end()) {
    is_dynamic = true;
  }
  if (is_dynamic && (ic == -1)) {
    ic = kc*groups;
    OP_LOGW(op.GetName().c_str(),
            "input x channel is unknow, fixed channel = %d, "
            "in_channels should be kc*grous[%d * %d]", (int)ic, (int)kc, (int)groups);
  }
  if ((!unknown_rank) && (groups == 1)) {
    if ((ic > 0) && (ic % kc == 0)) {
      groups = ic / kc;
      op.SetAttr("groups", groups);
      OP_LOGD(op.GetName().c_str(), "parameter groups is implicitly changed.");
    } else {
      OP_LOGE(op.GetName().c_str(), "in_channels(>0) should be divisible by kernel_channels when groups = 1.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "in_channels(>0) should be divisible by kernel_channels when groups = 1.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  }
  if ((!unknown_rank) && (ic != kc * groups)) {
    OP_LOGE(op.GetName().c_str(),
            "x channel should be equal to filter channel*groups. "
            "x format is: %s, filter format is: %s, "
            "x shape is: [%d,%d,%d,%d], filter shape is: [%d,%d,%d,%d], "
            "groups is: %d.",
            TypeUtils::FormatToSerialString(x_format).c_str(), TypeUtils::FormatToSerialString(w_format).c_str(),
            (int)x_shape[0], (int)x_shape[1], (int)x_shape[2], (int)x_shape[3], (int)w_shape[0], (int)w_shape[1],
            (int)w_shape[2], (int)w_shape[3], (int)groups);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["x_shape"] = std::to_string(x_shape[0]) + ", " + std::to_string(x_shape[1]) + ", " +
                         std::to_string(x_shape[2]) + ", " + std::to_string(x_shape[3]);
    err_map["filter_shape"] = std::to_string(w_shape[0]) + ", " + std::to_string(w_shape[1]) + ", " +
                              std::to_string(w_shape[2]) + ", " + std::to_string(w_shape[3]);
    err_map["groups"] = std::to_string(groups);

    std::string report_error_code = "E50059";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (kn % groups != 0) {
    OP_LOGE(op.GetName().c_str(), "out_channels should be divisible by groups.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "out_channels should be divisible by groups.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  if (!GetAttrsConv2D(op, x_format, strh, strw, dilh, dilw) ||
      !GetPadConv2D(op, ih, iw, kh, kw, strh, strw, dilh, dilw, padt, padb, padl, padr)) {
    return GRAPH_FAILED;
  }

  int64_t ihPad = (ih + padt + padb - dilh * (kh - 1) - 1);
  int64_t iwPad = (iw + padl + padr - dilw * (kw - 1) - 1);
  int64_t oh = ihPad / strh + 1;
  int64_t ow = iwPad / strw + 1;
  if (unknown_rank) {
    oh = -1;
    ow = -1;
  }
  if ((ih > 0) && (kh > 0) && (iw > 0) && (kw > 0)) {
    if ((ihPad < 0) || (iwPad < 0)) {
      OP_LOGE(op.GetName().c_str(), "image size after padding should be greater than or equal to filter size.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "image size after padding should be greater than or equal to filter size.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
  }

  vector<int64_t> y_shape;
  auto y_tensor = op_desc->MutableOutputDesc("y");
  auto y_format = y_tensor->GetFormat();
  CHECK_FORMAT(y_format)
  if (y_format == FORMAT_NCHW) {
    y_shape.push_back(in);
    y_shape.push_back(kn);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
  } else if (y_format == FORMAT_NHWC) {
    y_shape.push_back(in);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
    y_shape.push_back(kn);
  } else {
    OP_LOGE(op.GetName().c_str(),
            "output y format should be NCHW or NHWC."
            " actual is: %s",
            TypeUtils::FormatToSerialString(y_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "y";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_format_list"] = "NCHW or NHWC";
    err_map["format"] = TypeUtils::FormatToSerialString(y_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  y_tensor->SetShape(GeShape(y_shape));
  auto x_dtype = x_tensor->GetDataType();
  if (x_dtype == ge::DT_INT8) {
    y_tensor->SetDataType(ge::DT_INT32);
  } else {
    y_tensor->SetDataType(x_dtype);
  }

  // fuzz_build switch
  bool fuzz_build = false;
  op.GetAttr(ge::ATTR_NAME_FUZZ_BUILD, fuzz_build);

  // set Range
  if (is_dynamic) {
    OP_LOGD(op.GetName().c_str(), "start accurate build.");
    vector<int32_t> attr_params = {strh, strw,dilh, dilw,
                                   padt, padb, padl, padr,
                                   kn, kh, kw};
    if (!SetConv2dOutShapeRange(op, attr_params, y_shape, x_tensor, y_tensor)) {
      return GRAPH_FAILED;
    }
  }
  // fuzz build allow shape dim -1 with range
  if ((!unknown_rank) && fuzz_build) {
    OP_LOGD(op.GetName().c_str(), "start fuzz build.");
    // generate range
    std::vector<std::pair<int64_t, int64_t>> input_range;
    if (!GenConv2dShapeRange(op, x_tensor, input_range)){
      return GRAPH_FAILED;
    }
    // change pad to -1 when padding is SAME
    std::string pad_str;
    op.GetAttr("padding", pad_str);
    if (pad_str == "SAME") {
      op.SetAttr("pads", {-1, -1, -1, -1});
      OP_LOGD(op.GetName().c_str(), "set pads to {-1, -1, -1, -1} when padding is SAME in fuzzy build");
    }
    // only need to set input fuzz build range
    graphStatus ret = x_tensor->SetShapeRange(input_range);
    if (ret != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "set input range failed");
      return GRAPH_FAILED;
    }
    for (size_t i = 0; i < input_range.size(); i++) {
      OP_LOGD(op.GetName().c_str(), "input Range[%u] is (%lld, %lld)", i, input_range[i].first, input_range[i].second);
    }
  }
  OP_LOGD(op.GetName().c_str(), "Leave Conv2DInfer.");
  return GRAPH_SUCCESS;
}

/*!
  * @brief Verify the required 2 input tensor, optional bias ignored, verify
  *        strides and dilations attrs, pads ignored.
  * @param Conv2D Operator type.
  * @param Conv2DVerify Input validity check function.
  * @return Status The processing flow result.
  */
IMPLEMT_VERIFIER(Conv2D, Conv2DVerify) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv2DVerify.");
  auto x_tensor = op.GetInputDesc("x");
  auto w_tensor = op.GetInputDesc("filter");
  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();
  auto offset_w_tensor = op.GetInputDesc("offset_w");

  if (offset_w_tensor.GetOriginShape().GetDims().size() != 0) {
    OP_LOGE(op.GetName().c_str(), "input offset_w is not supported.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "input offset_w is not supported.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  bool unknown_rank = IsUnknownRankShape(x_shape);
  if ((!unknown_rank) && (x_shape.size() != 4)) {
    if (x_shape.size() == 0) {
      OP_LOGE(op.GetName().c_str(), "input x shape is empty.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input x shape is empty.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    } else {
      OP_LOGE(op.GetName().c_str(), "input x shape should be 4D.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input x shape should be 4D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
    return GRAPH_FAILED;
  }
  if (w_shape.size() != 4) {
    if (w_shape.size() == 0) {
      OP_LOGE(op.GetName().c_str(), "input filter shape is empty.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input filter shape is empty.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    } else {
      OP_LOGE(op.GetName().c_str(), "input filter shape should be 4D.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input filter shape should be 4D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
    return GRAPH_FAILED;
  }

  auto x_dtype = x_tensor.GetDataType();
  auto w_dtype = w_tensor.GetDataType();

  if (x_dtype != w_dtype) {
    OP_LOGE(op.GetName().c_str(),
            "input x dtype is differ from filter dtype."
            " actual x dtype is: %s filter dtype is: %s",
            TypeUtils::DataTypeToSerialString(x_dtype).c_str(),
            TypeUtils::DataTypeToSerialString(w_dtype).c_str());
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param1"] = "x";
    err_map["param1_data_type"] = TypeUtils::DataTypeToSerialString(x_dtype);
    err_map["param2"] = "filter";
    err_map["param2_data_type"] = TypeUtils::DataTypeToSerialString(w_dtype);
    err_map["rule"] = "input x dtype is same as filter dtype";
    std::string report_error_code = "E50004";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list)) {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "get strides list failed.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  std::vector<int32_t> dilation_list;
  if (GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)) {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "get dilations list failed.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "Leave Conv2DVerify.");
  return GRAPH_SUCCESS;
}

static void InferHWConv2D(int32_t input, int32_t kernel, int32_t pad, int32_t stride,
                          int32_t dilation, vector<int64_t> output_slice, vector<int64_t>& data_slice,
                          bool& start_add_pad, bool& end_add_pad) {
  // calc start rule: (i_start + pad_h)/stride_h = output_start
  int64_t i_start = output_slice[0] * stride - pad;
  if (i_start < 0) {
    start_add_pad = true;
    i_start = 0;
  }
  // calc end rule: (iend_start + pad_h)/stride_h = output_end
  // iend_end = iend_start + dilation*(kernel_h-1)
  int64_t i_end = output_slice[1] * stride - pad + dilation * (kernel - 1);
  if (i_end >= input) {
    end_add_pad = true;
    i_end = input - 1;
  }
  data_slice = {i_start, i_end};
}

/*!
  * @brief provide Conv2D operator slice data
  * @param Conv2D Operator type.
  * @param Conv2DInferDataSlice slice data function
  * @return Status The processing flow result.
  */
IMPLEMT_INFER_DATA_SLICE(Conv2D, Conv2DInferDataSlice) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv2D InferDataSlice");
  // get input h/w, filter h/w, pad_h,pad_w, stride_h, stride_w, dilation_h,dilation_w
  auto x_tensor = op.GetInputDesc("x");
  auto w_tensor = op.GetInputDesc("filter");

  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();

  auto x_format = x_tensor.GetOriginFormat();
  auto w_format = w_tensor.GetOriginFormat();

  std::vector<int32_t> stride_list;
  std::vector<int32_t> dilation_list;
  std::vector<int32_t> pad_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list) || GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)
      || GRAPH_SUCCESS != op.GetAttr("pads", pad_list)){
    return GRAPH_FAILED;
  }
  CHECK(pad_list.size() < 4, CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "pads size less then 4."),
    return GRAPH_FAILED);

  int32_t ih = 0;
  int32_t iw = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = pad_list[0];
  int32_t padb = pad_list[1];
  int32_t padl = pad_list[2];
  int32_t padr = pad_list[3];

  if (x_format == FORMAT_NCHW) {
    ih = x_shape[2];
    iw = x_shape[3];
    strh = stride_list[2];
    strw = stride_list[3];
    dilh = dilation_list[2];
    dilw = dilation_list[3];
  } else if (x_format == FORMAT_NHWC) {
    ih = x_shape[1];
    iw = x_shape[2];
    strh = stride_list[1];
    strw = stride_list[2];
    dilh = dilation_list[1];
    dilw = dilation_list[2];
  } else {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "x format is valid, the error x format is: %d", x_format);
    return GRAPH_FAILED;
  }

  if (w_format == FORMAT_NCHW) {
    kh = w_shape[2];
    kw = w_shape[3];
  } else if (w_format == FORMAT_NHWC) {
    kh = w_shape[1];
    kw = w_shape[2];
  } else if (w_format == FORMAT_HWCN) {
    kh = w_shape[0];
    kw = w_shape[1];
  } else {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "weight format is valid, the error w format is: %d", w_format);
    return GRAPH_FAILED;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }
  bool have_slice = false;
  vector<int> new_pad_lists;
  if (GRAPH_SUCCESS != op.GetAttr("pads", new_pad_lists)) {
    return GRAPH_FAILED;
  }
  for(int i=0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      have_slice = true;
      if (i == 2) {
        vector<int64_t> ih_slice;
        bool top_add_pad = false;
        bool bom_add_pad = false;
        InferHWConv2D(ih, kh, padt, strh, dilh, y_data_slice[i], ih_slice, top_add_pad, bom_add_pad);
        OP_LOGD(op.GetName().c_str(), "conv2d h axis slice ori_scope is [%d,%d], output scope is [%d,%d]",
                ih_slice[0], ih_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
        if (!top_add_pad) {
          new_pad_lists[0] = 0;
        }
        if (!bom_add_pad) {
          new_pad_lists[1] = 0;
        }
        x_data_slice[i] = ih_slice;
      } else if (i == 3) {
        vector<int64_t> iw_slice;
        bool left_add_pad = false;
        bool right_add_pad = false;
        InferHWConv2D(iw, kw, padl, strw, dilw, y_data_slice[i], iw_slice, left_add_pad, right_add_pad);
        OP_LOGD(op.GetName().c_str(), "conv2d w axis slice ori_scope is [%d,%d], output scope is [%d,%d]",
                iw_slice[0], iw_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
        if (!left_add_pad) {
          new_pad_lists[2] = 0;
        }
        if (!right_add_pad) {
          new_pad_lists[3] = 0;
        }
        x_data_slice[i] = iw_slice;
      } else {
        x_data_slice[i] = y_data_slice[i];
      }
    }
  }
  op.SetAttr("pads", new_pad_lists);
  OP_LOGD(op.GetName().c_str(), "conv2d new pad lists is [%d,%d,%d,%d]", new_pad_lists[0],
          new_pad_lists[1], new_pad_lists[2], new_pad_lists[3]);

  if (have_slice == false) {
    return GRAPH_FAILED;
  }
  if (!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
    return GRAPH_FAILED;
  }
  OP_LOGD(op.GetName().c_str(), "Calc Conv2D InferDataSlice end!");
  return GRAPH_SUCCESS;

}

INFER_DATA_SLICE_FUNC_REG(Conv2D, Conv2DInferDataSlice);
INFER_FUNC_REG(Conv2D, Conv2DInfer);
VERIFY_FUNC_REG(Conv2D, Conv2DVerify);
} // namespace ge