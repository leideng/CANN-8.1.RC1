/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_GRAPH_SERIALIZATION_UTILS_SERIALIZATION_UTIL_H_
#define METADEF_GRAPH_SERIALIZATION_UTILS_SERIALIZATION_UTIL_H_

#include "proto/ge_ir.pb.h"
#include "graph/types.h"

namespace ge {
class SerializationUtil {
 public:
  static void GeDataTypeToProto(const ge::DataType ge_type, proto::DataType &proto_type);
  static void ProtoDataTypeToGe(const proto::DataType proto_type, ge::DataType &ge_type);
 private:
  SerializationUtil() = delete;
};
}  // namespace ge

#endif // METADEF_GRAPH_SERIALIZATION_UTILS_SERIALIZATION_UTIL_H_
