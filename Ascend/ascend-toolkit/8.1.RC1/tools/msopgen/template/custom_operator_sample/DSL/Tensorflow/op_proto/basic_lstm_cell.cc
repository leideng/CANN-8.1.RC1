/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file basic_lstm_cell.cc
 * \brief
 */
#include "basic_lstm_cell.h"

#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "util/util.h"
#include "error_util.h"
#include "op_log.h"

namespace ge {
IMPLEMT_VERIFIER(BasicLSTMCell, BasicLSTMCellVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BasicLSTMCell, BasicLSTMCellInferShape) {
  ge::TensorDesc inputHTensorDesc = op.GetInputDesc("h");
  ge::TensorDesc inputCTensorDesc = op.GetInputDesc("c");
  ge::TensorDesc inputWTensorDesc = op.GetInputDesc("w");
  ge::Shape shape = inputCTensorDesc.GetShape();
  ge::Shape shapeH = inputHTensorDesc.GetShape();
  DataType inputHDtype = inputHTensorDesc.GetDataType();
  DataType inputCDtype = inputCTensorDesc.GetDataType();

  inputWTensorDesc.SetFormat(ge::FORMAT_HWCN);
  inputWTensorDesc.SetOriginFormat(ge::FORMAT_HWCN);

  TensorDesc outputCtTensorDesc = op.GetOutputDesc("ct");
  TensorDesc outputHtTensorDesc = op.GetOutputDesc("ht");
  TensorDesc outputItTensorDesc = op.GetOutputDesc("it");
  TensorDesc outputJtTensorDesc = op.GetOutputDesc("jt");
  TensorDesc outputFtTensorDesc = op.GetOutputDesc("ft");
  TensorDesc outputOtTensorDesc = op.GetOutputDesc("ot");
  TensorDesc outputTanhctTensorDesc = op.GetOutputDesc("tanhct");

  outputCtTensorDesc.SetShape(shape);
  outputCtTensorDesc.SetDataType(inputCDtype);
  outputHtTensorDesc.SetShape(shapeH);
  outputHtTensorDesc.SetDataType(inputHDtype);
  outputItTensorDesc.SetShape(shape);
  outputItTensorDesc.SetDataType(inputCDtype);
  outputJtTensorDesc.SetShape(shape);
  outputJtTensorDesc.SetDataType(inputCDtype);
  outputFtTensorDesc.SetShape(shape);
  outputFtTensorDesc.SetDataType(inputCDtype);
  outputOtTensorDesc.SetShape(shape);
  outputOtTensorDesc.SetDataType(inputCDtype);
  outputTanhctTensorDesc.SetShape(shape);
  outputTanhctTensorDesc.SetDataType(inputCDtype);

  (void)op.UpdateOutputDesc("ct", outputCtTensorDesc);
  (void)op.UpdateOutputDesc("ht", outputHtTensorDesc);
  (void)op.UpdateOutputDesc("it", outputItTensorDesc);
  (void)op.UpdateOutputDesc("jt", outputJtTensorDesc);
  (void)op.UpdateOutputDesc("ft", outputFtTensorDesc);
  (void)op.UpdateOutputDesc("ot", outputOtTensorDesc);
  (void)op.UpdateOutputDesc("tanhct", outputTanhctTensorDesc);
  (void)op.UpdateInputDesc("w", inputWTensorDesc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BasicLSTMCell, BasicLSTMCellInferShape);
VERIFY_FUNC_REG(BasicLSTMCell, BasicLSTMCellVerify);
// ----------------BasicLSTMCell Op-------------------
}  // namespace ge
