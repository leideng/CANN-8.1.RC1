/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "attr_serializer_registry.h"

#include "graph/serialization/bool_serializer.h"
#include "graph/serialization/buffer_serializer.h"
#include "graph/serialization/data_type_serializer.h"
#include "graph/serialization/float_serializer.h"
#include "graph/serialization/graph_serializer.h"
#include "graph/serialization/int_serializer.h"
#include "graph/serialization/list_list_float_serializer.h"
#include "graph/serialization/list_list_int_serializer.h"
#include "graph/serialization/list_value_serializer.h"
#include "graph/serialization/named_attrs_serializer.h"
#include "graph/serialization/string_serializer.h"
#include "graph/serialization/tensor_desc_serializer.h"
#include "graph/serialization/tensor_serializer.h"

#include "common/ge_common/debug/ge_log.h"
#include "graph/debug/ge_log.h"

namespace ge {
REG_GEIR_SERIALIZER(attr_bool, BoolSerializer, GetTypeId<bool>(), proto::AttrDef::kB);
REG_GEIR_SERIALIZER(attr_buffer, BufferSerializer, GetTypeId<ge::Buffer>(), proto::AttrDef::kBt);
REG_GEIR_SERIALIZER(attr_data_type, DataTypeSerializer, GetTypeId<ge::DataType>(), proto::AttrDef::kDt);
REG_GEIR_SERIALIZER(attr_float, FloatSerializer, GetTypeId<float>(), proto::AttrDef::kF);
REG_GEIR_SERIALIZER(attr_graph, GraphSerializer, GetTypeId<proto::GraphDef>(), proto::AttrDef::kG);
REG_GEIR_SERIALIZER(attr_int, IntSerializer, GetTypeId<int64_t>(), proto::AttrDef::kI);
REG_GEIR_SERIALIZER(attr_list, ListListFloatSerializer,
                    GetTypeId<std::vector<std::vector<float>>>(), proto::AttrDef::kListListFloat);
REG_GEIR_SERIALIZER(attr_list_list_int, ListListIntSerializer,
                    GetTypeId<std::vector<std::vector<int64_t>>>(), proto::AttrDef::kListListInt);
REG_GEIR_SERIALIZER(attr_list_int, ListValueSerializer, GetTypeId<std::vector<int64_t>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(attr_list_str, ListValueSerializer, GetTypeId<std::vector<std::string>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(attr_list_float, ListValueSerializer, GetTypeId<std::vector<float>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(attr_list_bool, ListValueSerializer, GetTypeId<std::vector<bool>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(attr_list_tensor_desc, ListValueSerializer,
                    GetTypeId<std::vector<GeTensorDesc>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(attr_list_tensor, ListValueSerializer, GetTypeId<std::vector<GeTensor>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(attr_list_buffer, ListValueSerializer, GetTypeId<std::vector<Buffer>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(attr_list_graph_def, ListValueSerializer,
                    GetTypeId<std::vector<proto::GraphDef>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(attr_list_named_attrs, ListValueSerializer,
                    GetTypeId<std::vector<ge::NamedAttrs>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(attr_list_data_type, ListValueSerializer,
                    GetTypeId<std::vector<ge::DataType>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(attr_named_attrs, NamedAttrsSerializer, GetTypeId<ge::NamedAttrs>(), proto::AttrDef::kFunc);
REG_GEIR_SERIALIZER(attr_str, StringSerializer, GetTypeId<std::string>(), proto::AttrDef::kS);
REG_GEIR_SERIALIZER(attr_tensor_desc, TensorDescSerializer, GetTypeId<GeTensorDesc>(), proto::AttrDef::kTd);
REG_GEIR_SERIALIZER(attr_tensor, TensorSerializer, GetTypeId<GeTensor>(), proto::AttrDef::kT);

AttrSerializerRegistry &AttrSerializerRegistry::GetInstance() {
  static AttrSerializerRegistry instance;
  return instance;
}

void AttrSerializerRegistry::RegisterGeIrAttrSerializer(const GeIrAttrSerializerBuilder& builder,
                                                        const TypeId obj_type,
                                                        const proto::AttrDef::ValueCase proto_type) {
  const std::lock_guard<std::mutex> lck_guard(mutex_);
  if (serializer_map_.count(obj_type) > 0U) {
    return;
  }
  std::unique_ptr<GeIrAttrSerializer> serializer = builder();
  serializer_map_[obj_type] = serializer.get();
  deserializer_map_[proto_type] = serializer.get();
  serializer_holder_.push_back(std::move(serializer));
}

GeIrAttrSerializer *AttrSerializerRegistry::GetSerializer(const TypeId obj_type) {
  const std::map<TypeId, GeIrAttrSerializer *>::const_iterator iter = serializer_map_.find(obj_type);
  if (iter == serializer_map_.cend()) {
    // print type
    REPORT_INNER_ERROR("E18888", "Serializer for type has not been registered");
    GELOGE(FAILED, "Serializer for type has not been registered");
    return nullptr;
  }
  return iter->second;
}

GeIrAttrSerializer *AttrSerializerRegistry::GetDeserializer(const proto::AttrDef::ValueCase proto_type) {
  const std::map<proto::AttrDef::ValueCase, GeIrAttrSerializer *>::const_iterator iter =
      deserializer_map_.find(proto_type);
  if (iter == deserializer_map_.cend()) {
    REPORT_INNER_ERROR("E18888",
                       "Deserializer for type [%d] has not been registered",
                       static_cast<int32_t>(proto_type));
    GELOGE(FAILED, "Deserializer for type [%d] has not been registered", static_cast<int32_t>(proto_type));
    return nullptr;
  }
  return iter->second;
}

AttrSerializerRegistrar::AttrSerializerRegistrar(const GeIrAttrSerializerBuilder builder,
                                                 const TypeId obj_type,
                                                 const proto::AttrDef::ValueCase proto_type) noexcept {
  if (builder == nullptr) {
    GELOGE(FAILED, "SerializerBuilder is nullptr.");
    return;
  }
  AttrSerializerRegistry::GetInstance().RegisterGeIrAttrSerializer(builder, obj_type, proto_type);
}
}
