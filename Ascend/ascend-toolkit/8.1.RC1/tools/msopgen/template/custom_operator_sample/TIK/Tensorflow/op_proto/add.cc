/*
 * Copyright (C)  2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file add.cc
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "./add.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include "util/util.h"
#include "util/common_shape_fns.h"
#include "util/array_ops_shape_fns.h"
#include "error_util.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/common_error_codes.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/error_manager/error_manager.h"
#include "op_log.h"
#include "register/infer_data_slice_registry.h"

namespace ge {

IMPLEMT_VERIFIER(Add, AddVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AddInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Add, AddInferShape);
VERIFY_FUNC_REG(Add, AddVerify);
}