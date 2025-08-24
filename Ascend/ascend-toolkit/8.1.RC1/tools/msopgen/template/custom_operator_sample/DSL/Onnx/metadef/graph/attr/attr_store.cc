/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "graph/attr_store.h"
#include "common/ge_common/debug/ge_log.h"
#include "attribute_group/attr_group_base.h"
#include "attribute_group/attr_group_serialize.h"
#include "checker.h"
namespace ge {
const AttrId constInvalidAttrId = GetAttrId(0xffffffffU, 0U);

AnyValue *AttrStore::GetOrCreateAnyValue(const AttrId attr_id) const {
  return const_cast<AnyValue *>(GetAnyValue(attr_id));
}
AnyValue *AttrStore::MutableAnyValue(const AttrId attr_id) const noexcept {
  return const_cast<AnyValue *>(GetAnyValue(attr_id));
}
const AnyValue *AttrStore::GetAnyValue(const AttrId attr_id) const noexcept {
  const auto attr_type = GetAttrType(attr_id);
  if (attr_type == static_cast<uint32_t>(AttrType::kAttrPredefinedInIr)) {
    return pre_defined_attrs_.GetAnyValue(GetSubAttrId(attr_id));
  } else if (attr_type == static_cast<uint32_t>(AttrType::kAttrGeneral)) {
    return nullptr;  // general不支持
  } else {
    // empty
  }
  return nullptr;
}
AttrStore AttrStore::Create(const size_t pre_defined_attr_count) {
  AttrStore as;
  as.pre_defined_attrs_.Resize(pre_defined_attr_count);
  return as;
}

AttrStore::AttrStore(AttrStore &&other) {
  names_to_id_.swap(other.names_to_id_);
  Swap(other);
  other_attrs_.Swap(other.other_attrs_);

  attrs_groups_ptr_ = std::move(other.attrs_groups_ptr_);
  other.attrs_groups_ptr_.clear();
}

AttrStore &AttrStore::operator=(AttrStore &&other) {
  if (this == &other) {
    return *this;
  }
  names_to_id_.swap(other.names_to_id_);
  Swap(other);
  other_attrs_.Swap(other.other_attrs_);

  for (auto &iter : attrs_groups_ptr_) {
    if (iter.second != nullptr) {
      iter.second.reset();
    }
  }
  attrs_groups_ptr_.clear();
  attrs_groups_ptr_ = std::move(other.attrs_groups_ptr_);
  other.attrs_groups_ptr_.clear();
  return *this;
}

void AttrStore::CopyAttrStoreAllMembers(const AttrStore &other) {
  names_to_id_ = other.names_to_id_;
  pre_defined_attrs_ = other.pre_defined_attrs_;
  general_attrs_ = other.general_attrs_;
  other_attrs_ = other.other_attrs_;

  for (auto &other_attrs_ptr : other.attrs_groups_ptr_) {
    if (other_attrs_ptr.second != nullptr) {
      auto attrs_group_ptr_ = other_attrs_ptr.second->Clone();
      if (attrs_group_ptr_ == nullptr) {
        GELOGE(GRAPH_FAILED, "Failed to alloc memory for attribute group.");
      }
      attrs_groups_ptr_.emplace(other_attrs_ptr.first, std::move(attrs_group_ptr_));
    }
  }
}

AttrStore::AttrStore(const AttrStore &other) {
  CopyAttrStoreAllMembers(other);
}

AttrStore &AttrStore::operator=(const AttrStore &other) {
  if (this == &other) {
    return *this;
  }
  for (auto &attrs_group_ptr_ : attrs_groups_ptr_) {
    if (attrs_group_ptr_.second != nullptr) {
      attrs_group_ptr_.second.reset();
    }
  }
  attrs_groups_ptr_.clear();
  CopyAttrStoreAllMembers(other);
  return *this;
}

const AnyValue *AttrStore::GetAnyValue(const std::string &name) const noexcept {
  const auto id = GetIdByName(name);
  if (id != constInvalidAttrId) {
    return pre_defined_attrs_.GetAnyValue(GetSubAttrId(id));
  }

  const AnyValue *const av = general_attrs_.GetAnyValue(name);
  if (av != nullptr) {
    return av;
  }

  return nullptr;
}
AnyValue *AttrStore::MutableAnyValue(const std::string &name) const noexcept {
  return const_cast<AnyValue *>(GetAnyValue(name));
}
AnyValue *AttrStore::GetOrCreateAnyValue(const std::string &name) {
  const auto id = GetIdByName(name);
  if (id != constInvalidAttrId) {
    return pre_defined_attrs_.GetOrCreateAnyValue(GetSubAttrId(id));
  }
  return general_attrs_.GetOrCreateAnyValue(name);
}
AttrId AttrStore::GetIdByName(const std::string &name) const noexcept {
  const auto iter = names_to_id_.find(name);
  if (iter == names_to_id_.end()) {
    return constInvalidAttrId;
  }
  return iter->second;
}
void AttrStore::SetNameAndId(std::string name, const AttrId id) {
  names_to_id_[std::move(name)] = id;
}
bool AttrStore::Exists(const AttrId attr_id) const noexcept {
  return GetAnyValue(attr_id) != nullptr;
}
bool AttrStore::Exists(const std::string &name) const noexcept {
  return GetAnyValue(name) != nullptr;
}
bool AttrStore::Delete(const std::string &name) {
  const auto iter = names_to_id_.find(name);
  if (iter != names_to_id_.end()) {
    const auto sub_id = GetSubAttrId(iter->second);
    (void) names_to_id_.erase(iter);
    return pre_defined_attrs_.Delete(sub_id);
  }
  return general_attrs_.Delete(name);
}
std::set<std::string> AttrStore::GetAllAttrNames() const {
  std::set<std::string> names;
  for (const auto &iter : names_to_id_) {
    (void) names.insert(iter.first);
  }
  general_attrs_.GetAllNames(names);
  return names;
}
std::map<std::string, AnyValue> AttrStore::GetAllAttrs() const {
  return GetAllAttrsWithFilter(nullptr);
}
std::map<std::string, AnyValue> AttrStore::GetAllAttrsWithFilter(const AttrNameFilter &attr_filter) const {
  std::map<std::string, AnyValue> attrs;
  for (const auto &iter : names_to_id_) {
    const auto av = pre_defined_attrs_.GetAnyValue(GetSubAttrId(iter.second));
    if (av == nullptr) {
      // error
      continue;
    }
    if (av->IsEmpty()) {
      continue;
    }
    if ((attr_filter != nullptr) && (!attr_filter(iter.first))) {
      continue;
    }
    attrs[iter.first] = *av;
  }
  general_attrs_.GetAllAttrsWithFilter(attrs, attr_filter);
  return attrs;
}
void AttrStore::Swap(AttrStore &other) {
  pre_defined_attrs_.Swap(other.pre_defined_attrs_);
  general_attrs_.Swap(other.general_attrs_);
}

void AttrStore::PreDefinedAttrStore::Resize(const size_t s) {
  attrs_.resize(s);
}
bool AttrStore::PreDefinedAttrStore::Exists(const AttrSubId index) const noexcept {
  if (index >= attrs_.size()) {
    return false;
  }
  return !attrs_[static_cast<size_t>(index)].IsEmpty();
}
bool AttrStore::PreDefinedAttrStore::Delete(const AttrSubId index) {
  if (!Exists(index)) {
    return false;
  }
  attrs_[static_cast<size_t>(index)].Clear();
  return true;
}
AnyValue *AttrStore::PreDefinedAttrStore::GetOrCreateAnyValue(const AttrSubId index) const {
  return const_cast<AnyValue *>(GetAnyValue(index));
}
AnyValue *AttrStore::PreDefinedAttrStore::MutableAnyValue(const AttrSubId index) const noexcept {
  return const_cast<AnyValue *>(GetAnyValue(index));
}
const AnyValue *AttrStore::PreDefinedAttrStore::GetAnyValue(const AttrSubId index) const noexcept {
  if (index >= attrs_.size()) {
    return nullptr;
  }
  return &attrs_[static_cast<size_t>(index)];
}
void AttrStore::PreDefinedAttrStore::Swap(AttrStore::PreDefinedAttrStore &other) {
  attrs_.swap(other.attrs_);
}
bool AttrStore::CustomDefinedAttrStore::Exists(const std::string &name) const noexcept {
  return attrs_.count(name) > 0UL;
}
bool AttrStore::CustomDefinedAttrStore::Delete(const std::string &name) {
  return attrs_.erase(name) == 1UL;
}
AnyValue *AttrStore::CustomDefinedAttrStore::GetOrCreateAnyValue(const std::string &name) {
  return &attrs_[name];
}
AnyValue *AttrStore::CustomDefinedAttrStore::MutableAnyValue(const std::string &name) const noexcept {
  return const_cast<AnyValue *>(GetAnyValue(name));
}
const AnyValue *AttrStore::CustomDefinedAttrStore::GetAnyValue(const std::string &name) const noexcept {
  const auto iter = attrs_.find(name);
  if (iter != attrs_.end()) {
    return &iter->second;
  } else {
    return nullptr;
  }
}
void AttrStore::CustomDefinedAttrStore::GetAllNames(std::set<std::string> &names) const {
  for (const auto &iter : attrs_) {
    (void) names.insert(iter.first);
  }
}
void AttrStore::CustomDefinedAttrStore::GetAllAttrs(std::map<std::string, AnyValue> &names_to_attr) const {
  for (const auto &iter : attrs_) {
    names_to_attr[iter.first] = iter.second;
  }
}
void AttrStore::CustomDefinedAttrStore::GetAllAttrsWithFilter(std::map<std::string, AnyValue> &names_to_attr,
                                                              const AttrNameFilter &attr_filter) const {
  for (const auto &iter : attrs_) {
    if ((attr_filter != nullptr) && (!attr_filter(iter.first))) {
      continue;
    }
    names_to_attr[iter.first] = iter.second;
  }
}
void AttrStore::CustomDefinedAttrStore::Swap(AttrStore::CustomDefinedAttrStore &other) {
  attrs_.swap(other.attrs_);
}
bool AttrStore::SetAnyValueByName(const std::string &name, const AnyValue &value) {
  const auto av = GetOrCreateAnyValue(name);
  if (av == nullptr) {
    return false;
  }
  *av = value;
  return true;
}
void AttrStore::Clear() {
  pre_defined_attrs_.Clear();
  general_attrs_.Clear();
}
void AttrStore::PreDefinedAttrStore::Clear() {
  attrs_.clear();
}
void AttrStore::CustomDefinedAttrStore::Clear() {
  attrs_.clear();
}

graphStatus AttrStore::SetAttrToOtherGroup(const std::string &attr, const AnyValue &value) {
  return other_attrs_.SetAttr(attr, value);
}

graphStatus AttrStore::GetAttrFromOtherGroup(const std::string &attr, AnyValue &value) const {
  return other_attrs_.GetAttr(attr, value);
}

const std::unordered_map<std::string, AnyValue> &AttrStore::FastGetAllAttrsFromOtherGroup() const {
  return other_attrs_.FastGetAllAttr();
}

std::unordered_map<std::string, AnyValue> AttrStore::GetAllAttrsFromOtherGroup() const {
  return other_attrs_.GetAllAttr();
}

bool AttrStore::CheckAttrIsExistInOtherGroup(const std::string &attr) const {
  return other_attrs_.CheckAttrIsExist(attr);
}

bool AttrStore::DeleteSingleAttrsInOtherGroup(const std::string &attr) {
  return other_attrs_.DeleteSingleAttr(attr);
}

void AttrStore::DeleteAllAttrsInOtherGroup() {
  other_attrs_.DeleteAllAttrs();
}

const AttrGroupsMap& AttrStore::GetAttrsGroupPtr() const {
  return attrs_groups_ptr_;
}

void AttrStore::ClearAllAttrs() {
  other_attrs_.DeleteAllAttrs();
  for (auto &attrs_group_ptr_ : attrs_groups_ptr_) {
    if (attrs_group_ptr_.second != nullptr) {
      attrs_group_ptr_.second.reset();
    }
  }
  attrs_groups_ptr_.clear();
}

void AttrStore::ClearAllAttrsInOtherAttrs() {
  return other_attrs_.DeleteAllAttrs();
}

bool AttrStore::ClearAttrInOtherAttrs(const std::string &attr_name) {
  return other_attrs_.DeleteSingleAttr(attr_name);
}

const std::unordered_set<std::string> OtherAttrs::valid_attrs_ = {"Max memory"};

bool OtherAttrs::CheckAttrIsValid(const std::string &attr) const {
  if (valid_attrs_.find(attr) != valid_attrs_.end()) {
    return true;
  }
  return false;
}

graphStatus OtherAttrs::SetAttr(const std::string &attr, const AnyValue &value) {
  if (CheckAttrIsValid(attr)) {
    keys_to_attrs_[attr] = value;
    return GRAPH_SUCCESS;
  }

  REPORT_INNER_ERROR("E18888", "Failed to set the %s.", attr.c_str());
  GELOGE(ge::GRAPH_FAILED, "Failed to set the %s.", attr.c_str());
  return GRAPH_FAILED;
}

graphStatus OtherAttrs::GetAttr(const std::string &attr, AnyValue &value) const {
  auto iter = keys_to_attrs_.find(attr);
  if (iter != keys_to_attrs_.end()) {
    value = iter->second;
    return GRAPH_SUCCESS;
  }

  REPORT_INNER_ERROR("E18888", "Failed to find the %s.", attr.c_str());
  GELOGE(ge::GRAPH_FAILED, "Failed to find the %s.", attr.c_str());
  return GRAPH_FAILED;
}

const std::unordered_map<std::string, AnyValue> &OtherAttrs::FastGetAllAttr() const {
  return keys_to_attrs_;
}

std::unordered_map<std::string, AnyValue> OtherAttrs::GetAllAttr() const {
  return keys_to_attrs_;
}

bool OtherAttrs::CheckAttrIsExist(const std::string &attr) const {
  auto iter = keys_to_attrs_.find(attr);
  if (iter != keys_to_attrs_.end()) {
    return true;
  }

  return false;
}

bool OtherAttrs::DeleteSingleAttr(const std::string &attr) {
  auto iter = keys_to_attrs_.find(attr);
  if (iter != keys_to_attrs_.end()) {
    keys_to_attrs_.erase(iter);
    return true;
  }

  return false;
}

void OtherAttrs::DeleteAllAttrs() {
  keys_to_attrs_.clear();
}

void OtherAttrs::Swap(OtherAttrs &other) {
  keys_to_attrs_.swap(other.keys_to_attrs_);
}
graphStatus AttrGroupSerialize::SerializeAllAttr(proto::AttrGroups &attr_groups, const AttrStore &attr_store) {
  GE_ASSERT_GRAPH_SUCCESS(OtherGroupSerialize(attr_groups, attr_store));

  auto& id_2_ptr = attr_store.GetAttrsGroupPtr();
  for (const auto& ptr : id_2_ptr) {
    if (ptr.second != nullptr) {
      GE_ASSERT_GRAPH_SUCCESS(ptr.second->Serialize(*attr_groups.add_attr_group_def()));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus AttrGroupSerialize::DeserializeAllAttr(const proto::AttrGroups &attr_group, AttrStore &attr_store) {
  GE_ASSERT_GRAPH_SUCCESS(OtherGroupDeserialize(attr_group, attr_store));
  for (const auto &attr_group_def : attr_group.attr_group_def()) {
    switch (attr_group_def.attr_group_case()) {
      case proto::AttrGroupDef::AttrGroupCase::kOpAttrGroup:
        break;
        // todo: 需要后续考虑属性组的反序列化放在哪个so里, graph_base依赖了属性组反序列化
      default:
        GELOGW("Unsupported AttrGroupCase %d, Deserialize Attr group failed.",
               static_cast<int32_t>(attr_group_def.attr_group_case()));
        return GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}
// todo: other group计划是需要再主线替换掉当前Ge IR上的的map<string, AttrDef> attr = 5字段
//       这个在分支上暂时不需要切换，等主线切换后再做替换，当前先做属性组的序列化和反序列化
graphStatus AttrGroupSerialize::OtherGroupSerialize(proto::AttrGroups &attr_groups, const AttrStore &attr_store) {
  (void)attr_store;
  (void)attr_groups;
  return GRAPH_SUCCESS;
}

graphStatus AttrGroupSerialize::OtherGroupDeserialize(const proto::AttrGroups &attr_groups, AttrStore &attr_store) {
  (void)attr_store;
  (void)attr_groups;
  return GRAPH_SUCCESS;
}
}  // namespace ge
