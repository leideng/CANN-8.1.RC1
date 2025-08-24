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
 * \file conv3d.cc
 * \brief
 */
#define CHECK_POSITION(position)                                                       \
  {                                                                                    \
    if (position < 0) {                                                                \
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "get position failed:%s:%d", #position, position); \
      return false;                                                                    \
    }                                                                                  \
  }
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

#include "./conv3d.h"

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
  map<int, std::string> format2str = {
    {ge::FORMAT_NCHW, "NCHW"}, {ge::FORMAT_NHWC, "NHWC"}, {ge::FORMAT_HWCN, "HWCN"},
    {ge::FORMAT_DHWNC, "DHWNC"}, {ge::FORMAT_DHWCN, "DHWCN"}, {ge::FORMAT_NDHWC, "NDHWC"},
    {ge::FORMAT_NCDHW, "NCDHW"}
  };
  map<int, std::string> dtype2str = {
    {ge::DT_FLOAT, "FLOAT"}, {ge::DT_FLOAT16, "FLOAT16"}, {ge::DT_INT8, "INT8"},
    {ge::DT_INT16, "INT16"}, {ge::DT_UINT16, "UINT16"}, {ge::DT_UINT8, "UINT8"},
    {ge::DT_INT32, "INT32"}, {ge::DT_INT64, "INT64"}, {ge::DT_UINT32, "UINT32"},
    {ge::DT_UINT64, "UINT64"}
  };
}
// ---------------------------Conv3D---------------------------
template <typename T1>
static bool CheckVectorAnyNegative(const std::vector<T1>& list)
{
    for (const auto& iter : list) {
        if (iter < 0) {
            return false;
        }
    }
    return true;
}

static bool VerifyConv3dDilations(const ge::Operator& op, std::vector<int32_t>& dilation_list) {
  //check dilations shape
  if (op.GetAttr("dilations", dilation_list) != GRAPH_SUCCESS) {
    dilation_list.clear();
    for (int32_t i = 0; i < kConv3dDimSizeLimit; i++) {
      dilation_list.push_back(1);
    }
    OP_LOGI(op.GetName().c_str(), "no dilations setting, use dilations as [1,1,1,1,1]");
  }
  auto d_size = dilation_list.size();
  if (d_size != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "dilations list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dilation_list";
    err_map["op_name"] = "Conv3d or Conv3dbp or Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
    err_map["input_value"] = std::to_string(d_size);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}

static bool GetPadConv3D(ge::Operator& op, int32_t id, int32_t ih, int32_t iw,
                         int32_t kd, int32_t kh, int32_t kw, int32_t strd,
                         int32_t strh, int32_t strw, int32_t dild, const int32_t dilh,
                         int32_t dilw, int32_t& padf, int32_t& padba, int32_t& padt,
                         int32_t& padb, int32_t& padl, int32_t& padr) {
  std::string pad_str;
  std::vector<int32_t> pad_list;
  if (GRAPH_SUCCESS == op.GetAttr("_padding", pad_list)) {
    op.SetAttr("pads", pad_list);
  } else if (GRAPH_SUCCESS == op.GetAttr("padding", pad_str)) {
    if (pad_str.compare("SAME") == 0) {
      int32_t tails_d = id % strd;
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t dilate_kernel_d = dild * (kd - 1) + 1;
      int32_t dilate_kernel_h = dilh * (kh - 1) + 1;
      int32_t dilate_kernel_w = dilw * (kw - 1) + 1;
      int32_t pad_d = std::max((tails_d > 0 ? dilate_kernel_d - tails_d : dilate_kernel_d - strd), 0);
      int32_t pad_h = std::max((tails_h > 0 ? dilate_kernel_h - tails_h : dilate_kernel_h - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? dilate_kernel_w - tails_w : dilate_kernel_w - strw), 0);
      pad_list.push_back(pad_d / 2);
      pad_list.push_back(pad_d / 2 + pad_d % 2);
      pad_list.push_back(pad_h / 2);
      pad_list.push_back(pad_h / 2 + pad_h % 2);
      pad_list.push_back(pad_w / 2);
      pad_list.push_back(pad_w / 2 + pad_w % 2);
    } else if (pad_str.compare("VALID") == 0) {
      for (int32_t i = 0; i < 6; i++)
        pad_list.push_back(0);
    } else {
      OP_LOGE(op.GetName().c_str(), "padding should be SAME or VALID.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "padding";
      err_map["op_name"] = "Conv3d";
      err_map["Expected_value"] = "SAME or VALID";
      err_map["input_value"] = pad_str;
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    op.SetAttr("pads", pad_list);
  }
  std::vector<int32_t> pad_vec;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pad_vec)) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "get pads failed");
    return false;
  }
  auto p_size = pad_vec.size();
  if (p_size != kConv3dPadsSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "pads list should be 6d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "pads_list";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "6d";
    err_map["input_value"] = std::to_string(p_size);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  padf = pad_vec[0];
  padba = pad_vec[1];
  padt = pad_vec[2];
  padb = pad_vec[3];
  padl = pad_vec[4];
  padr = pad_vec[5];

  auto x_shape = op.GetInputDesc("x").GetShape().GetDims();
  bool unknown_rank = IsUnknownRankShape(x_shape);
  bool unknown_shape = IsUnKnownShape(x_shape);
  bool negative_pad = (padf < 0 || padba < 0 || padt < 0 || padb < 0 || padl < 0 || padr < 0);
  // dynamic shape pad maybe negative
  if ((!unknown_shape) && (!unknown_rank) && negative_pad) {
    OP_LOGE(op.GetName().c_str(), "pads should be positive");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "pads_list";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "positive";
    err_map["input_value"] = std::to_string(padf) + " " + \
                             std::to_string(padba) + " " + \
                             std::to_string(padt) + " " + \
                             std::to_string(padb) + " " + \
                             std::to_string(padl) + " " + \
                             std::to_string(padr);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

static bool GetAttrsConv3D(ge::Operator& op, Format refer,  int32_t& strd,
                           int32_t& strh, int32_t& strw, int32_t& dild,
                           int32_t& dilh, int32_t& dilw) {
  std::vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "get strides list failed.");
    return false;
  }
  auto s_size = stride_list.size();
  if (s_size != kConv3dStridesSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "strides list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "stride_list";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "5d";
    err_map["input_value"] = std::to_string(s_size);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "no data format setting, using NDHWC");
    data_format = FORMAT_NDHWC;
  }

  std::vector<int32_t> dilation_list;
  if (!VerifyConv3dDilations(op, dilation_list)) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "get dilation attrs failed.");
    return false;
  }

  if (refer == FORMAT_NCDHW) {
    strd = stride_list[2];
    strh = stride_list[3];
    strw = stride_list[4];
    dild = dilation_list[2];
    dilh = dilation_list[3];
    dilw = dilation_list[4];
  } else if (refer == FORMAT_NDHWC) {
    strd = stride_list[1];
    strh = stride_list[2];
    strw = stride_list[3];
    dild = dilation_list[1];
    dilh = dilation_list[2];
    dilw = dilation_list[3];
  }
  if (strd <= 0 || strh <= 0 || strw <= 0) {
    OP_LOGE(op.GetName().c_str(), "strides should be positive.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "positive";
    err_map["input_value"] = std::to_string(strd) + " " + std::to_string(strh) + " " + std::to_string(strw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (dild != 1) {
    OP_LOGE(op.GetName().c_str(), "dilations in the D dimension only supports 1 now.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "1";
    err_map["input_value"] = std::to_string(dild);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

static void SetConv3dOutShapeDimRange(const std::string& padding,
                                      size_t idx,
                                      map<std::string, int32_t>& attr_params,
                                      const std::vector<std::pair<int64_t, int64_t>>& fm_range,
                                      std::vector<std::pair<int64_t, int64_t>>& out_range) {
  int32_t stride = attr_params["stride"];
  int32_t dilation = attr_params["dilation"];
  int32_t pad = attr_params["pad"];
  int32_t kernel = attr_params["kernel"];
  int64_t low = fm_range[idx].first;
  int64_t high = fm_range[idx].second;
  if (padding == "SAME") {
    out_range[idx].first = (low + stride - 1) / stride;
    out_range[idx].second = (high + stride - 1) / stride;
  } else {
    out_range[idx].first = (low + pad - dilation * (kernel - 1) - 1) / stride + 1;
    out_range[idx].second = (high + pad - dilation * (kernel - 1) - 1) / stride + 1;
  }

  out_range[idx].first = std::max(out_range[idx].first, kDynamicRangeLowerBound);
  if (high == -1) {
    out_range[idx].second = high;
  } else {
    out_range[idx].second = std::min(out_range[idx].second, kDynamicRangeUpperBound);
  }
}

static bool IsDHWUnknown(const string& op_name, const string& tensor_name, const vector<int64_t>& shape, Format format) {
  size_t idx_n = DIM_INDEX0;
  size_t idx_c = DIM_INDEX4;
  if (format == FORMAT_NCDHW) {
    idx_c = DIM_INDEX1;
  }

  vector<int64_t> shape_copy = shape;
  if (shape.size() > idx_n) {
    shape_copy[idx_n] = 1;
  }

  if ((shape.size() > idx_c) && (shape[idx_c] == -1)) {
    OP_LOGW(op_name.c_str(), "input %s channel is unknown", tensor_name.c_str());
    shape_copy[idx_c] = 1;
  }

  return IsUnKnownShape(shape_copy);
}

static bool SetConv3dOutShapeRange(op::Conv3D& op,
                                   map<std::string, int32_t>& attr_params,
                                   vector<int64_t>& y_shape,
                                   TensorDesc& y_tensor) {
  auto x_tensor = op.get_input_desc_x();
  auto x_shape = x_tensor.GetShape().GetDims();
  bool unknown_rank = IsUnknownRankShape(x_shape);
  bool unknown_shape = IsUnKnownShape(x_shape);

  // default format: NDHWC
  size_t idx_n = DIM_INDEX0;
  size_t idx_d = DIM_INDEX1;
  size_t idx_h = DIM_INDEX2;
  size_t idx_w = DIM_INDEX3;
  size_t idx_c = DIM_INDEX4;
  if (op.get_input_desc_x().GetFormat() == FORMAT_NCDHW) {
    idx_c = DIM_INDEX1;
    idx_d = DIM_INDEX2;
    idx_h = DIM_INDEX3;
    idx_w = DIM_INDEX4;
  }

  // update pads if padding is SAME
  std::string padding;
  // when rank is unknown, or D/H/W is unknown, set SAME padding as -1
  if (IsUnknownRankShape(x_shape) || IsDHWUnknown(op.GetName(), "x", x_shape, x_tensor.GetFormat())) {
    std::string pad_str;
    if (op.GetAttr("padding", pad_str) == GRAPH_SUCCESS) {
      std::vector<int32_t> pads(kConv3dPadsSizeLimit, 0);
      if (pad_str == "SAME") {
        pads.assign(kConv3dPadsSizeLimit, -1);
        OP_LOGD(op.GetName().c_str(), "set pads to {-1, -1, -1, -1, -1, -1} when padding is SAME in dynamic_shape");
      }
      op.SetAttr("pads", pads);
    }
  }

  if (!unknown_shape) {
    return true;
  }

  int32_t strd = attr_params["strd"];
  int32_t strh = attr_params["strh"];
  int32_t strw = attr_params["strw"];
  int32_t dild = attr_params["dild"];
  int32_t dilh = attr_params["dilh"];
  int32_t dilw = attr_params["dilw"];
  int32_t padf = attr_params["padf"];
  int32_t padba = attr_params["padba"];
  int32_t padt = attr_params["padt"];
  int32_t padb = attr_params["padb"];
  int32_t padl = attr_params["padl"];
  int32_t padr = attr_params["padr"];
  int32_t kn = attr_params["kn"];
  int32_t kd = attr_params["kd"];
  int32_t kh = attr_params["kh"];
  int32_t kw = attr_params["kw"];

  OP_LOGD(op.GetName().c_str(), "dynamic shape set range");
  std::vector<std::pair<int64_t, int64_t>> fm_range;
  x_tensor.GetShapeRange(fm_range);
  if (fm_range.empty()) {
    OP_LOGW(op.GetName().c_str(), "fm_range's shape is empty.");
    if (x_shape[idx_d] == -1) {
      y_shape[idx_d] = -1;
    }

    if (x_shape[idx_h] == -1) {
      y_shape[idx_h] = -1;
    }

    if (x_shape[idx_w] == -1) {
      y_shape[idx_w] = -1;
    }
    // op will check this invalid range
    return true;
  }

  if (fm_range.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "fm_range's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "fm_range";
    err_map["op_name"] = "Conv3DInfer";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(fm_range.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<std::pair<int64_t, int64_t>> out_range(fm_range);
  out_range[idx_c] = std::make_pair(static_cast<int64_t>(kn), static_cast<int64_t>(kn));
  if (x_shape[idx_d] == -1) {
    y_shape[idx_d] = -1;
    map<std::string, int32_t> attr_params_d = {
      {"stride", strd}, {"dilation", dild},
      {"pad", padf + padba}, {"kernel", kd}
    };
    // attr_params_d data structure should keep same as SetConv3dOutShapeDimRange
    SetConv3dOutShapeDimRange(padding, idx_d, attr_params_d, fm_range, out_range);
  }

  if (x_shape[idx_h] == -1) {
    y_shape[idx_h] = -1;
    map<std::string, int32_t> attr_params_h = {
      {"stride", strh}, {"dilation", dilh},
      {"pad", padt + padb}, {"kernel", kh}
    };
    // attr_params_h data structure should keep same as SetConv3dOutShapeDimRange
    SetConv3dOutShapeDimRange(padding, idx_h, attr_params_h, fm_range, out_range);
  }

  if (x_shape[idx_w] == -1) {
    y_shape[idx_w] = -1;
    map<std::string, int32_t> attr_params_w = {
      {"stride", strw}, {"dilation", dilw},
      {"pad", padl + padr}, {"kernel", kw}
    };
    // attr_params_w data structure should keep same as SetConv3dOutShapeDimRange
    SetConv3dOutShapeDimRange(padding, idx_w, attr_params_w, fm_range, out_range);
  }

  y_tensor.SetShape(Shape(y_shape));
  y_tensor.SetShapeRange(out_range);

  return true;
}

static bool NormalizeConv3dShape(const op::Conv3D& op, vector<int64_t>& x_shape_new, vector<int64_t>& w_shape_new) {
  auto x_tensor = op.get_input_desc_x();
  auto x_format = x_tensor.GetFormat();
  auto x_shape = x_tensor.GetShape().GetDims();
  bool unknown_rank = IsUnknownRankShape(x_shape);
  if (!((x_shape.size() == kConv3dInputSizeLimit) || unknown_rank)) {
    OP_LOGE(op.GetName().c_str(), "x_shape's shape should be 5d or -2.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "x_shape";
    err_map["op_name"] = "Conv3DInfer";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(x_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  int32_t in = -1;
  int32_t ic = -1;
  int32_t id = -1;
  int32_t ih = -1;
  int32_t iw = -1;
  if (x_format == FORMAT_NCDHW) {
    if (!unknown_rank) {
      in = x_shape[DIM_INDEX0];
      ic = x_shape[DIM_INDEX1];
      id = x_shape[DIM_INDEX2];
      ih = x_shape[DIM_INDEX3];
      iw = x_shape[DIM_INDEX4];
    }
  } else if (x_format == FORMAT_NDHWC) {
    if (!unknown_rank) {
      in = x_shape[DIM_INDEX0];
      ic = x_shape[DIM_INDEX4];
      id = x_shape[DIM_INDEX1];
      ih = x_shape[DIM_INDEX2];
      iw = x_shape[DIM_INDEX3];
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "input x format should be NCDHW or NDHWC.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "x_format";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "NCDHW or NDHWC";
    err_map["input_value"] = x_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  auto w_tensor = op.get_input_desc_filter();
  auto w_shape = w_tensor.GetShape().GetDims();
  auto w_format = w_tensor.GetFormat();
  if (w_shape.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "w_shape's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "w_shape";
    err_map["op_name"] = "Conv3DInfer";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(w_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  int32_t kn = 0;
  int32_t kc = 0;
  int32_t kd = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  if (w_format == FORMAT_NCDHW) {
    kn = w_shape[DIM_INDEX0];
    kc = w_shape[DIM_INDEX1];
    kd = w_shape[DIM_INDEX2];
    kh = w_shape[DIM_INDEX3];
    kw = w_shape[DIM_INDEX4];
  } else if (w_format == FORMAT_NDHWC) {
    kn = w_shape[DIM_INDEX0];
    kc = w_shape[DIM_INDEX4];
    kd = w_shape[DIM_INDEX1];
    kh = w_shape[DIM_INDEX2];
    kw = w_shape[DIM_INDEX3];
  } else if (w_format == FORMAT_DHWCN) {
    kn = w_shape[DIM_INDEX4];
    kc = w_shape[DIM_INDEX3];
    kd = w_shape[DIM_INDEX0];
    kh = w_shape[DIM_INDEX1];
    kw = w_shape[DIM_INDEX2];
  } else {
    OP_LOGE(op.GetName().c_str(), "input filter format should be NCDHW, NDHWC or DHWCN.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "wFormat";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "NCDHW or NDHWC or DHWCN";
    err_map["input_value"] = w_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  x_shape_new.clear();
  x_shape_new.push_back(in);
  x_shape_new.push_back(id);
  x_shape_new.push_back(ih);
  x_shape_new.push_back(iw);
  x_shape_new.push_back(ic);

  w_shape_new.clear();
  w_shape_new.push_back(kn);
  w_shape_new.push_back(kd);
  w_shape_new.push_back(kh);
  w_shape_new.push_back(kw);
  w_shape_new.push_back(kc);

  return true;
}

IMPLEMT_INFERFUNC(Conv3D, Conv3DInfer) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv3DInfer.");

  auto x_tensor = op.get_input_desc_x();
  auto x_format = x_tensor.GetFormat();
  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_tensor = op.get_input_desc_filter();

  bool unknown_rank = IsUnknownRankShape(x_shape);
  vector<int64_t> x_shape_new;
  vector<int64_t> w_shape_new;
  if (!NormalizeConv3dShape(op, x_shape_new, w_shape_new)) {
      return GRAPH_FAILED;
  }

  int32_t in = x_shape_new[DIM_INDEX0];
  int32_t id = x_shape_new[DIM_INDEX1];
  int32_t ih = x_shape_new[DIM_INDEX2];
  int32_t iw = x_shape_new[DIM_INDEX3];
  int32_t ic = x_shape_new[DIM_INDEX4];

  int32_t kn = w_shape_new[DIM_INDEX0];
  int32_t kd = w_shape_new[DIM_INDEX1];
  int32_t kh = w_shape_new[DIM_INDEX2];
  int32_t kw = w_shape_new[DIM_INDEX3];
  int32_t kc = w_shape_new[DIM_INDEX4];

  int64_t group = 1;
  if (GRAPH_SUCCESS != op.GetAttr("groups", group)) {
    OP_LOGI(op.GetName().c_str(), "no group setting, use group as 1");
  }

  if (ic == -1) {
    // print warn in IsDHWUnknown later
    ic = kc * group;
  }

  if ((!unknown_rank) && (ic != kc * group)) {
    OP_LOGE(op.GetName().c_str(), "input x channel should be equal to filter.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3d";
    err_map["channel_of_x"] = std::to_string(ic);
    err_map["channel_of_filter"] = std::to_string(kc * group);
    std::string report_error_code = "E50039";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dild = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  if (!GetAttrsConv3D(op, x_format, strd, strh, strw, dild, dilh, dilw)) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "get attrs failed.");
    return GRAPH_FAILED;
  }

  int32_t padf = 0;
  int32_t padba = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  if (!GetPadConv3D(op, id, ih, iw, kd, kh, kw, strd, strh, strw, dild, dilh, dilw, padf, padba, padt, padb,
                            padl, padr)) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "get pads attrs failed.");
    return GRAPH_FAILED;
  }

  int64_t od = (id + padf + padba - dild * (kd - 1) - 1) / strd + 1;
  int64_t oh = (ih + padt + padb - dilh * (kh - 1) - 1) / strh + 1;
  int64_t ow = (iw + padl + padr - dilw * (kw - 1) - 1) / strw + 1;
  if (unknown_rank) {
      od = -1;
      oh = -1;
      ow = -1;
  }

  vector<int64_t> y_shape;
  auto y_tensor = op.get_output_desc_y();
  auto y_format = y_tensor.GetFormat();

  if (y_format == FORMAT_NCDHW) {
    y_shape.push_back(in);
    y_shape.push_back(kn);
    y_shape.push_back(od);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
  } else if (y_format == FORMAT_NDHWC) {
    y_shape.push_back(in);
    y_shape.push_back(od);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
    y_shape.push_back(kn);
  } else {
    OP_LOGE(op.GetName().c_str(), "output y format should be NCDHW or NDHWC.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "yFormat";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "NCDHW or NDHWC";
    err_map["input_value"] = y_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  y_tensor.SetShape(Shape(y_shape));
  auto x_dtype = x_tensor.GetDataType();
  y_tensor.SetDataType(x_dtype);

  // set dynamic out range
  map<std::string, int32_t> attr_params = {
    {"strd", strd}, {"strh", strh}, {"strw", strw},
    {"dild", dild}, {"dilh", dilh}, {"dilw", dilw},
    {"padf", padf}, {"padba", padba}, {"padt", padt},
    {"padb", padb}, {"padl", padl}, {"padr", padr},
    {"kn", kn}, {"kd", kd}, {"kh", kh}, {"kw", kw}
  };
  // attr_params data structure should keep same as SetConv3dOutShapeRange
  // GE will convert y_shape to only one -2 if shape contains -2, so can't get y_shape via y_tensor.GetShape
  if (!SetConv3dOutShapeRange(op, attr_params, y_shape, y_tensor)) {
    return GRAPH_FAILED;
  }

  if (GRAPH_SUCCESS != op.update_output_desc_y(y_tensor)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "output_desc_y";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = GRAPH_SUCCESS;
    err_map["output_value"] = op.update_output_desc_y(y_tensor);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  OP_LOGD(op.GetName().c_str(), "Leave Conv3DInfer.");
  return GRAPH_SUCCESS;
}

static void InferHWConv3d(int32_t kernel,
                          int32_t dilation,
                          int32_t stride,
                          int32_t input_size,
                          const vector<int64_t>& output,
                          vector<int64_t>& input,
                          vector<int32_t>& pad_list,
                          uint32_t pad_idx) {
  int32_t kernel_size = (kernel - 1) * dilation + 1;
  int32_t pad_h = pad_list[pad_idx];
  input[0] = std::max(stride * output[0] - pad_h, 0L);
  input[1] = std::min(stride * output[1] - pad_h + kernel_size - 1,
                      static_cast<int64_t>(input_size - 1));

  pad_list[pad_idx] = std::max(static_cast<int32_t>(pad_h - stride * output[0]), 0);
  pad_list[pad_idx + 1] = std::max(static_cast<int32_t>(
                                      stride * output[1] - pad_h +
                                      kernel_size - input_size),
                                   0);
}

IMPLEMT_INFER_DATA_SLICE(Conv3D, Conv3DInferSliceData) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv3DInferSliceData.");

  auto x_format = op.get_input_desc_x().GetFormat();
  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dild = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  if (!GetAttrsConv3D(op, x_format, strd, strh, strw, dild, dilh, dilw)) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "get attrs failed.");
    return GRAPH_FAILED;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");

  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}, {}};
  vector<vector<int64_t>> w_data_slice = {{}, {}, {}, {}};
  vector<vector<int64_t>> bias_data_slice = {{}};

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "no data slice, need not infer input.");
    return GRAPH_FAILED;
  }

  // check data_slice attr
  if(y_data_slice.size() != kConv3dDataSlice) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "y_data_slice's size should be 6.");
    return GRAPH_FAILED;
  }

  // no support for C0 axis
  if(y_data_slice[kConv3dDataSlice - 1].size() != 0) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "no support for cut C0 axis.");
    return NOT_SUPPORT_SLICE;
  }

  // check valid slice num in data slice
  int32_t valid_cnt = 0;
  for(uint32_t i = 0; i < y_data_slice.size(); ++i) {
    if(y_data_slice[i].size() == 0) {
      continue;
    }
    if(y_data_slice[i].size() != 2) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "data slice format input size should be 2.");
      return GRAPH_FAILED;
    }
    valid_cnt ++;
  }
  if(valid_cnt == 0) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "data slice is empty.");
    return GRAPH_FAILED;
  }
  if(valid_cnt != 1) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "valid slice range num is more than 1.");
    return GRAPH_FAILED;
  }

  vector<int32_t> pad_list;
  op.GetAttr("pads", pad_list);
  bool needUpdateX = false;
  bool needUpdateW = false;

  auto x_shape = op.get_input_desc_x().GetShape().GetDims();
  std::string x_format_str = format2str[x_format];
  int32_t d_input_position = x_format_str.find("D");
  CHECK_POSITION(d_input_position);
  int32_t h_input_position = x_format_str.find("H");
  CHECK_POSITION(h_input_position);
  int32_t w_input_position = x_format_str.find("W");
  CHECK_POSITION(w_input_position);
  int32_t id = x_shape[d_input_position];
  int32_t ih = x_shape[h_input_position];
  int32_t iw = x_shape[w_input_position];

  auto filter_format = op.get_input_desc_filter().GetFormat();
  auto w_shape = op.get_input_desc_filter().GetShape().GetDims();
  std::string filter_format_str = format2str[filter_format];
  int32_t d_filter_position = filter_format_str.find("D");
  CHECK_POSITION(d_filter_position);
  int32_t h_filter_position = filter_format_str.find("H");
  CHECK_POSITION(h_filter_position);
  int32_t w_filter_position = filter_format_str.find("W");
  CHECK_POSITION(w_filter_position);
  int32_t kd = w_shape[d_filter_position];
  int32_t kh = w_shape[h_filter_position];
  int32_t kw = w_shape[w_filter_position];

  // cut N
  if(y_data_slice[0].size() != 0) {
    x_data_slice[0] = y_data_slice[0];
    needUpdateX = true;
  }

  // cut D
  if(y_data_slice[1].size() != 0) {
    x_data_slice[1].clear();
    x_data_slice[1].resize(2);
    InferHWConv3d(kd, dild, strd, id,
                  y_data_slice[1], x_data_slice[1], pad_list, 0);
    needUpdateX = true;
  }

  // cut Cout
  if(y_data_slice[2].size() != 0) {
    w_data_slice[1] = y_data_slice[2];
    bias_data_slice[0] = y_data_slice[2];
    needUpdateW = true;
  }

  // cut H
  if(y_data_slice[3].size() != 0) {
    x_data_slice[3].clear();
    x_data_slice[3].resize(2);
    InferHWConv3d(kh, dilh, strh, ih,
                  y_data_slice[3], x_data_slice[3], pad_list, 2);
    needUpdateX = true;
  }

  // cut W
  if(y_data_slice[4].size() != 0) {
    x_data_slice[4].clear();
    x_data_slice[4].resize(2);
    InferHWConv3d(kw, dilw, strw, iw,
                  y_data_slice[4], x_data_slice[4], pad_list, 4);
    needUpdateX = true;
  }

  // check update flag
  if(!needUpdateX && !needUpdateW) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "there's no update in desc.");
    return GRAPH_FAILED;
  }

  // update data slice attr
  if(needUpdateX) {
    if(!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "set x data slice attr failed.");
      return GRAPH_FAILED;
    }
  }
  if(needUpdateW){
    if(!AttrUtils::SetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice)) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "set w data slice attr failed");
      return GRAPH_FAILED;
    }
  }

  // update pads attr info
  op.SetAttr("pads", pad_list);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3D, Conv3DVerify) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv3DVerify.");
  auto x_tensor = op.GetInputDesc("x");
  auto w_tensor = op.GetInputDesc("filter");

  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();
  bool unknown_rank = IsUnknownRankShape(x_shape);
  if (!((x_shape.size() == kConv3dInputSizeLimit) || unknown_rank)) {
    OP_LOGE(op.GetName().c_str(), "input x shape should be 5d or -2.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "xShape_size";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = std::to_string(5);
    err_map["output_value"] = std::to_string(x_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (w_shape.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "input filter shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "w_shape_size";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = std::to_string(5);
    err_map["output_value"] = std::to_string(w_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto x_dtype = x_tensor.GetDataType();
  auto w_dtype = w_tensor.GetDataType();

  if (x_dtype != w_dtype) {
    OP_LOGE(op.GetName().c_str(), "input x dtype is differ from filter dtype.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3d";
    err_map["attr_name"] = "dtype";
    err_map["param1_name"] = "input x";
    err_map["param2_name"] = "weight";
    err_map["param1_value"] = std::to_string(x_dtype);
    err_map["param2_value"] = std::to_string(w_dtype);
    std::string report_error_code = "E50031";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list)) {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3d";
    err_map["op_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (stride_list.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "strides list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  std::vector<int32_t> dilation_list;
  if (!VerifyConv3dDilations(op, dilation_list)) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "get dilation attrs failed.");
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "Leave Conv3DVerify.");
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Conv3D, Conv3DInferSliceData);
INFER_FUNC_REG(Conv3D, Conv3DInfer);
VERIFY_FUNC_REG(Conv3D, Conv3DVerify);
} // namespace ge