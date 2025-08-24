/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file conv3d_plugin.cpp
 * \brief
 */
#include <map>

#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"
#include "common/util/error_manager/error_manager.h"

#include "op_log.h"

namespace domi {

namespace {
  const int kPos0 = 0;
  const int kPos1 = 1;
}

Status ParseParamsConv3D(const Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDesc org_tensor_w = op_dsc->GetInputDesc(kPos1);
  org_tensor_w.SetOriginFormat(ge::FORMAT_DHWCN);
  org_tensor_w.SetFormat(ge::FORMAT_DHWCN);
  auto ret = op_dsc->UpdateInputDesc(kPos1, org_tensor_w);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update filter format failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3d";
    err_map["param_name"] = "updating filter's format";
    err_map["rule_desc"] = "update filter's format";
    err_map["format"] = "failed";
    std::string report_error_code = "E50012";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "update filter format success, now is %d", op.GetInputDesc(kPos1).GetFormat());

  ge::Format data_format = ge::FORMAT_NDHWC;
  std::string data_format_attr;
  if (op.GetAttr("data_format", data_format_attr) == ge::GRAPH_SUCCESS) {
    if (data_format_attr == "NCDHW") {
      data_format = ge::FORMAT_NCDHW;
    }
  }

  ge::GeTensorDesc org_tensor_x = op_dsc->GetInputDesc(kPos0);
  org_tensor_x.SetOriginFormat(data_format);
  org_tensor_x.SetFormat(data_format);
  ret = op_dsc->UpdateInputDesc(kPos0, org_tensor_x);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update input x format failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3d";
    err_map["param_name"] = "updating input_x's format";
    err_map["rule_desc"] = "update input_x's format";
    err_map["format"] = "failed";
    std::string report_error_code = "E50012";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "update input x format success, now is %d", op.GetInputDesc(kPos0).GetFormat());

  ge::GeTensorDesc org_tensor_y = op_dsc->GetOutputDesc(kPos0);
  org_tensor_y.SetOriginFormat(data_format);
  org_tensor_y.SetFormat(data_format);
  ret = op_dsc->UpdateOutputDesc(kPos0, org_tensor_y);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update output y format failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3d";
    err_map["param_name"] = "updating output_y's format";
    err_map["rule_desc"] = "update output_y's format";
    err_map["format"] = "failed";
    std::string report_error_code = "E50012";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return FAILED;
  }
  std::vector<int32_t> pad_list = {0, 0, 0, 0, 0, 0};
  op.SetAttr("pads", pad_list);

  OP_LOGI(op.GetName().c_str(), "update output y format success, now is %d", op.GetInputDesc(kPos0).GetFormat());
  return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv3D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv3D")
    .ParseParamsFn(ParseParamsConv3D)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
