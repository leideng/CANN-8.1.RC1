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
 * \file conv2d_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "external/graph/operator.h"
#include "common/util/error_manager/error_manager.h"

#include "op_log.h"

namespace domi {

const int kInputX = 0;
const int kInputFilter = 1;
const size_t kPaddingSize = 8;

/*!
  * @brief Replace GE ParseParams fuction to process graph conv2d node attrs
  * @param op_src the source op info from tf.
  * @param op the dest GE op.
  * @return status whether this operation success.
  */
Status ParseParamsConv2D(const Message* op_src, ge::Operator& op) {
  // Convert original tf graph conv2d attrs to GE graph attrs
  if (AutoMappingFn(op_src, op) != SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "auto mapping failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "auto mapping failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return FAILED;
  }

  // The filter format shuold be HWCN, not NHWC or NCHW, so set here to fix this problem
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_dsc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "get op desc failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "get op desc failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return FAILED;
  }
  ge::GeTensorDesc org_tensor_w = op_dsc->GetInputDesc(kInputFilter);
  org_tensor_w.SetOriginFormat(ge::FORMAT_HWCN);
  org_tensor_w.SetFormat(ge::FORMAT_HWCN);
  auto ret = op_dsc->UpdateInputDesc(kInputFilter, org_tensor_w);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update filter format failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "update filter format failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return FAILED;
  }

  int32_t pad_top = 0;
  int32_t pad_bottom = 0;
  int32_t pad_left = 0;
  int32_t pad_right = 0;
  // String type padding is processed during infershape
  std::vector<int32_t> padding_list;
  if (op.GetAttr("explicit_paddings", padding_list) == ge::GRAPH_SUCCESS && padding_list.size() == kPaddingSize) {
    ge::GeTensorDesc org_tensor_x = op_dsc->GetInputDesc(kInputX);
    auto x_format = org_tensor_x.GetOriginFormat();
    if (x_format == ge::FORMAT_NCHW) {
      pad_top = padding_list[4];
      pad_bottom = padding_list[5];
      pad_left = padding_list[6];
      pad_right = padding_list[7];
    } else if (x_format == ge::FORMAT_NHWC) {
      pad_top = padding_list[2];
      pad_bottom = padding_list[3];
      pad_left = padding_list[4];
      pad_right = padding_list[5];
    }
  }

  // Escape GE require attr [pads] check here
  std::vector<int32_t> pad_list = {pad_top, pad_bottom, pad_left, pad_right};
  op.SetAttr("pads", pad_list);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv2D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv2D")
    .ParseParamsFn(ParseParamsConv2D)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
