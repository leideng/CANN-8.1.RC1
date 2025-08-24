/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "float_serializer.h"
#include "proto/ge_ir.pb.h"
#include "graph/debug/ge_log.h"
#include "graph/types.h"

namespace ge {
graphStatus FloatSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  float32_t val;
  const graphStatus ret = av.GetValue(val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get float attr.");
    return GRAPH_FAILED;
  }
  def.set_f(val);
  return GRAPH_SUCCESS;
}

graphStatus FloatSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  return av.SetValue(def.f());
}

REG_GEIR_SERIALIZER(float_serializer, FloatSerializer, GetTypeId<float>(), proto::AttrDef::kF);
}  // namespace ge
