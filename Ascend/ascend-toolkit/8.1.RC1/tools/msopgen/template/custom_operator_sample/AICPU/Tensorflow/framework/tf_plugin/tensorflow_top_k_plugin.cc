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
 * \file top_k_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "op_log.h"

namespace domi {
Status TopKMappingFn(const Message* op_src, ge::Operator& op) {
  if (AutoMappingFn(op_src, op) != SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "AutoMappingFn failed.");
    return FAILED;
  }
  int32_t dim = -1;
  op.SetAttr("dim", dim);
  bool largest = true;
  op.SetAttr("largest", largest);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("TopK")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TopKV2")
    .ParseParamsFn(TopKMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
