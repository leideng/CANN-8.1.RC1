/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "register/tilingdata_base.h"
#include <cstring>
#include <securec.h>
#ifndef ASCENDC_DEVICE_REG_STATIC
#include "common/ge_common/debug/ge_log.h"
#endif
#include "graph/ascend_string.h"

namespace optiling {
std::vector<FieldInfo> TilingDef::GetFieldInfo() const {
  return field_info_;
}

const char *TilingDef::GetTilingClassName() const {
  return class_name_;
}

size_t TilingDef::GetDataSize() const {
  return data_size_;
}

bool CheckPathIsHeader(std::string file) {
  const std::string suffix = ".h";
  if (suffix.size() > file.size()) {
    return false;
  }
  return file.substr(file.size() - suffix.size()) == suffix;
}

const char* GetFileName(const char* path) {
  const char* file_name = strrchr(path, '/');
  if (!file_name) {
    return path;
  } else {
    file_name++;
    return file_name;
  }
}

uint32_t __attribute__((weak)) TilingDataStructBase::RecordTilingStruct(const char* name, const char* file, \
    uint32_t line) {
  const char* file_name = GetFileName(file);
  bool is_header = CheckPathIsHeader(std::string(file_name));
  auto it = records.find(name);
  if (it != records.end()) {
    std::pair<const char *, uint32_t> item = it->second;
    if (!is_header) {
      if (item.second != line) {
          printf("[Warning]: tiling struct [%s] is conflict with one in file %s, line %d\n", \
              name, item.first, item.second);
      }
      return 0;
    }
    if (!CheckPathIsHeader(std::string(item.first))) {
      return 0;
    }
    if ((strcmp(item.first, file_name) == 0) && item.second == line) {
      return 0;
    }
    printf("[Warning]: tiling struct [%s] is conflict with one in file %s, line %d\n", \
        name, item.first, item.second);
  } else {
    records.emplace(name, std::make_pair(file_name, line));
  }
  return 0;
}

void TilingDef::GeLogError(const std::string &str) const {
#ifndef ASCENDC_DEVICE_REG_STATIC
  GELOGE(ge::GRAPH_FAILED, "%s", str.c_str());
#endif
}

void TilingDef::SetDataPtr(void *dataPtr) {
  if (!inited_data_ptr && data_ptr_ != nullptr) {
    delete[] data_ptr_;
  }
  inited_data_ptr = true;
  data_ptr_ = (uint8_t*)dataPtr;
  for (auto &ptr : saveBufferPtr) {
    TilingDef* sub_ptr = (TilingDef *)ptr.first;
    size_t offset = ptr.second;
    uint8_t* struct_ptr = data_ptr_ + offset;
    sub_ptr->SetDataPtr(struct_ptr);
  }
}

void TilingDef::SaveToBuffer(void *pdata, size_t capacity) {
  if (inited_data_ptr) {
#ifndef ASCENDC_DEVICE_REG_STATIC
    GELOGD("TilingDef::SaveToBuffer, op %s, data had been saved.", class_name_);
#endif
    return;
  }
  // copy tilingdata to buffer without struct tiling data.
  auto mem_ret = memcpy_s(pdata, capacity, data_ptr_, data_size_);
  if (mem_ret != EOK) {
#ifndef ASCENDC_DEVICE_REG_STATIC
    GELOGE(ge::GRAPH_FAILED,
           "TilingDef::SaveToBuffer failed: memcpy_s return op [%s] [%d], capacity = [%zu], data_size_ = [%zu].",
           class_name_, mem_ret, capacity, data_size_);
#endif
  }
}

void TilingDef::CheckAlignAndGenPlaceHolder(const char *name, size_t typeSize) {
  if (data_size_ % typeSize == 0) {
    return;
  }
  size_t alignSize = typeSize - (data_size_ % typeSize);
  field_info_.emplace_back(FieldInfo("uint8_t", name, alignSize));
  data_size_ += alignSize;
  return;
}

void TilingDef::InitData() {
#ifndef ASCENDC_DEVICE_REG_STATIC
    GELOGD("TilingDef::InitData, op %s, data size %d.", class_name_, data_size_);
#endif
    data_ptr_ = new (std::nothrow)uint8_t[data_size_]();
    if (data_ptr_ == nullptr) {
#ifndef ASCENDC_DEVICE_REG_STATIC
          GELOGE(ge::GRAPH_FAILED, "TilingDef::InitData failed: op %s, init data size %d.", class_name_, data_size_);
#endif
          return;
    }
    for (auto &ptr : saveBufferPtr) {
      TilingDef* sub_ptr = (TilingDef *)ptr.first;
      size_t offset = ptr.second;
      uint8_t* struct_ptr = data_ptr_ + offset;
      sub_ptr->SetDataPtr(struct_ptr);
    }
}

CTilingDataClassFactory &CTilingDataClassFactory::GetInstance()
{
  static CTilingDataClassFactory instance;
  return instance;
}

void CTilingDataClassFactory::RegisterTilingData(const char *op_type,
                                                 const TilingDataConstructor constructor) {
  instance_.emplace(op_type, constructor);
#ifndef ASCENDC_DEVICE_REG_STATIC
  GELOGD("op_type: %s, registered count: %zu.", op_type, instance_.size());
#endif
}

std::shared_ptr<TilingDef> CTilingDataClassFactory::CreateTilingDataInstance(const char *op_type) {
  const auto it = instance_.find(op_type);
  if (it == instance_.end()) {
#ifndef ASCENDC_DEVICE_REG_STATIC
    GELOGW("cannot find op_type:%s.", op_type);
#endif
    return nullptr;
  }

  const TilingDataConstructor constructor = it->second;
  if (constructor == nullptr) {
#ifndef ASCENDC_DEVICE_REG_STATIC
    GELOGW("CreateTilingDataInstance: constructor is nullptr.");
#endif
    return nullptr;
  }

  return (*constructor)();
}
}  // end of namespace optiling
