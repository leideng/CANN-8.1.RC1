/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_REGISTER_INFER_DATA_SLICE_REGISTRY_H_
#define INC_REGISTER_INFER_DATA_SLICE_REGISTRY_H_

#include "external/graph/ge_error_codes.h"
#include "external/graph/operator.h"
#include "external/graph/types.h"

namespace ge {
using InferDataSliceFunc = std::function<graphStatus(Operator &)>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferDataSliceFuncRegister {
 public:
  InferDataSliceFuncRegister(const char_t *const operator_type, const InferDataSliceFunc &infer_data_slice_func);
  ~InferDataSliceFuncRegister() = default;
};
}  // namespace ge

// infer data slice func register
#define IMPLEMT_COMMON_INFER_DATA_SLICE(func_name) \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static graphStatus (func_name)(Operator &op)

#define IMPLEMT_INFER_DATA_SLICE(op_name, func_name) \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static graphStatus (func_name)(op::op_name &op)

#define INFER_DATA_SLICE_FUNC(op_name, x) [](Operator &v) { return (x)((op::op_name &)v); }

#define INFER_DATA_SLICE_FUNC_REG_IMPL(op_name, x, n) \
  static const InferDataSliceFuncRegister PASTE(ids_register, n)(#op_name, (x))

#define INFER_DATA_SLICE_FUNC_REG(op_name, x) \
  INFER_DATA_SLICE_FUNC_REG_IMPL(op_name, INFER_DATA_SLICE_FUNC(op_name, x), __COUNTER__)

#endif  // INC_REGISTER_INFER_DATA_SLICE_REGISTRY_H_
