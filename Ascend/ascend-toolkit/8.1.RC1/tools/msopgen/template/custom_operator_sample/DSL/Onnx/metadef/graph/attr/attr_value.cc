/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "external/graph/attr_value.h"
#include "debug/ge_util.h"
#include "graph/ge_attr_value.h"

#define ATTR_VALUE_SET_GET_IMP(type)                 \
  graphStatus AttrValue::GetValue(type &val) const { \
    if (impl != nullptr) {                           \
      return impl->geAttrValue_.GetValue<type>(val); \
    }                                                \
    return GRAPH_FAILED;                             \
  }

namespace ge {
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AttrValue::AttrValue() {
  impl = ComGraphMakeShared<AttrValueImpl>();
}

ATTR_VALUE_SET_GET_IMP(AttrValue::STR)
ATTR_VALUE_SET_GET_IMP(AttrValue::INT)
ATTR_VALUE_SET_GET_IMP(AttrValue::FLOAT)

graphStatus AttrValue::GetValue(AscendString &val) {
  std::string val_get;
  const auto status = GetValue(val_get);
  if (status != GRAPH_SUCCESS) {
    return status;
  }
  val = AscendString(val_get.c_str());
  return GRAPH_SUCCESS;
}
}  // namespace ge
