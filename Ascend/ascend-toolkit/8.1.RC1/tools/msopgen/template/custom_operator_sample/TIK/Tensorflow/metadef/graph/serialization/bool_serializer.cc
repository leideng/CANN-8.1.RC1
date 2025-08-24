/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "bool_serializer.h"
#include <string>
#include "proto/ge_ir.pb.h"
#include "graph/debug/ge_log.h"

namespace ge {
graphStatus BoolSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  bool val;
  const graphStatus ret = av.GetValue(val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get bool attr.");
    return GRAPH_FAILED;
  }
  def.set_b(val);
  return GRAPH_SUCCESS;
}

graphStatus BoolSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  return av.SetValue(def.b());
}

REG_GEIR_SERIALIZER(bool_serializer, BoolSerializer, GetTypeId<bool>(), proto::AttrDef::kB);
}  // namespace ge
