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
 * \file reduce_std.cpp
 * \brief
 */
#include "apply_adam_d.h"
#include "op_log.h"
#include "util/util.h"
#include "error_util.h"

namespace ge {
// Check input and attr of the input tensor description.
bool ApplyVerifyFunc(const ge::Operator& op, const std::vector<std::string>& inputTensorList,
                     const std::vector<std::string>& inputScalarList) {
  // check shape of Tensor
  auto var_dims = op.GetInputDesc(inputTensorList[0]).GetShape().GetDims();
  if (var_dims.size() > 8 || var_dims.size() < 0) {
    OP_LOGE(op.GetName().c_str(), "var only support 0 ~ 8 dims!");
    return GRAPH_FAILED;
  }
  if (IsUnknown(var_dims)) {
    OP_LOGW(op.GetName().c_str(), "this is dynamic shape, will exit ApplyVerifyFunc");
    return true;
  }
  for (std::size_t i = 1; i < inputTensorList.size(); i++) {
    auto tmp_dims = op.GetInputDesc(inputTensorList[i]).GetShape().GetDims();
    if (IsUnknown(tmp_dims)) {
      OP_LOGW(op.GetName().c_str(), "this is dynamic shape, will continue ApplyVerifyFunc");
      continue;
    }
    if (tmp_dims != var_dims) {
      OP_LOGE(op.GetName().c_str(), "the shape of %s must equal with %s", inputTensorList[i].c_str(),
              inputTensorList[0].c_str());
      return false;
    }
  }

  // check shape of Scalar
  for (std::size_t j = 0; j < inputScalarList.size(); j++) {
    auto scalar_dims = op.GetInputDesc(inputScalarList[j]).GetShape().GetDims();
    if (scalar_dims.size() > 1) {
      OP_LOGE(op.GetName().c_str(), "The input %s must be scalar!", inputScalarList[j].c_str());
      return false;
    }
  }
  return true;
}

// ----------------ApplyAdamD Op-------------------
IMPLEMT_VERIFIER(ApplyAdamD, ApplyAdamDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdamD proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("m");
  inputTensorList.push_back("v");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("beta1_power");
  inputScalarList.push_back("beta2_power");
  inputScalarList.push_back("lr");
  inputScalarList.push_back("beta1");
  inputScalarList.push_back("beta2");
  inputScalarList.push_back("epsilon");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

}  // namespace ge