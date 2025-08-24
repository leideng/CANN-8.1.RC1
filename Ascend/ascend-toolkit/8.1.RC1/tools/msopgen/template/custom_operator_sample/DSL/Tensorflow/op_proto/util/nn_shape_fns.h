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
 * \file nn_shape_fns.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_UTIL_NN_SHAPE_FNS_H_
#define OPS_BUILT_IN_OP_PROTO_UTIL_NN_SHAPE_FNS_H_

#include "graph/operator.h"
#include "graph/debug/ge_log.h"
#include "error_util.h"

namespace ge {
#define UNCHANGED_SHAPE()                                      \
  TensorDesc outputDesc = op.GetOutputDesc("y");               \
  outputDesc.SetShape(op.GetInputDesc(0).GetShape());          \
  outputDesc.SetDataType(op.GetInputDesc(0).GetDataType());    \
  outputDesc.SetFormat(FORMAT_NCHW);                           \
  if (op.UpdateOutputDesc("y", outputDesc) != GRAPH_SUCCESS) { \
    std::string err_msg = UpdateParamErrMsg("output y"); \
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg); \
    return GRAPH_FAILED;                                       \
  }                                                            \
  return GRAPH_SUCCESS;
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_UTIL_NN_SHAPE_FNS_H_
