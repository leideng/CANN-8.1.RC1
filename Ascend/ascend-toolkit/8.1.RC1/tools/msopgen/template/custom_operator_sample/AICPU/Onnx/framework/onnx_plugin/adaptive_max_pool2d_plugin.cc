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
 * \file adaptive_max_pool2d__plugin.cpp
 * \brief
 */
#include <string>
#include <vector>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {

Status ParseParamsAdaptiveMaxPool2d(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("AdaptiveMaxPool2d", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  std::vector<int> v_output_size = {};
  bool set_output_size_flag = false;

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "output_size" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints_size() == 2) {
        for (int i = 0; i < attr.ints_size(); i++) {
          v_output_size.push_back(attr.ints(i));
          }
        } else {
          OP_LOGE("AdaptiveMaxPool2d", "length of output_size must be 2.");
        }
      set_output_size_flag = true;
    }
  }

  if (set_output_size_flag) {
    op_dest.SetAttr("output_size", v_output_size);
  } else {
    OP_LOGE("AdaptiveMaxPool2d", "onnx AdaptiveMaxPool2d op has no output_size attr.");
  }
  return SUCCESS;
}

// register AdaptiveMaxPool2d op info to GE
REGISTER_CUSTOM_OP("AdaptiveMaxPool2d")
    .FrameworkType(ONNX)
    .OriginOpType("aten::adaptive_max_pool2d")
    .ParseParamsFn(ParseParamsAdaptiveMaxPool2d)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
