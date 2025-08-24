/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "tensor_serializer.h"
#include "proto/ge_ir.pb.h"
#include "graph/debug/ge_util.h"
#include "graph/debug/ge_log.h"
#include "tensor_desc_serializer.h"
#include "graph/ge_tensor.h"

namespace ge {
graphStatus TensorSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  GeTensor ge_tensor;
  const graphStatus ret = av.GetValue(ge_tensor);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get tensor attr.");
    return GRAPH_FAILED;
  }

  GeTensorSerializeUtils::GeTensorAsProto(ge_tensor, def.mutable_t());
  return GRAPH_SUCCESS;
}

graphStatus TensorSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  GeTensor ge_tensor;
  GeTensorSerializeUtils::AssembleGeTensorFromProto(&def.t(), ge_tensor);
  return av.SetValue(std::move(ge_tensor));
}

REG_GEIR_SERIALIZER(tesnor_serializer, TensorSerializer, GetTypeId<GeTensor>(), proto::AttrDef::kT);
}  // namespace ge
