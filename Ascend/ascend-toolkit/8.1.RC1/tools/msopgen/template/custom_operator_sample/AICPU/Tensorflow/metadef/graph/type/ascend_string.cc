/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "external/graph/ascend_string.h"
#include "debug/ge_log.h"
#include "common/util/mem_utils.h"

namespace ge {
AscendString::AscendString(const char_t *const name) {
  if (name != nullptr) {
    name_ = MakeShared<std::string>(name);
    if (name_ == nullptr) {
      REPORT_CALL_ERROR("E18888", "new string failed.");
      GELOGE(FAILED, "[New][String]AscendString[%s] make shared failed.", name);
    }
  }
}

AscendString::AscendString(const char_t *const name, size_t length) {
  if (name != nullptr) {
    name_ = MakeShared<std::string>(name, length);
    if (name_ == nullptr) {
      REPORT_CALL_ERROR("E18888", "new string with length failed.");
      GELOGE(FAILED, "[New][String]AscendString make shared failed, length=%zu.", length);
    }
  }
}

const char_t *AscendString::GetString() const {
  if (name_ == nullptr) {
    const static char *empty_value = "";
    return empty_value;
  }

  return (*name_).c_str();
}

size_t AscendString::GetLength() const {
  if (name_ == nullptr) {
    return 0UL;
  }

  return (*name_).length();
}

size_t AscendString::Hash() const {
  if (name_ == nullptr) {
    const static size_t kEmptyStringHash = std::hash<std::string>()("");
    return kEmptyStringHash;
  }

  return std::hash<std::string>()(*name_);
}

bool AscendString::operator<(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  } else {
    return (*name_) < (*(d.name_));
  }
}

bool AscendString::operator>(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return false;
  } else if (d.name_ == nullptr) {
    return true;
  } else {
    return (*name_) > (*(d.name_));
  }
}

bool AscendString::operator==(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return true;
  } else if (name_ == nullptr) {
    return false;
  } else if (d.name_ == nullptr) {
    return false;
  } else {
    return (*name_) == (*(d.name_));
  }
}

bool AscendString::operator<=(const AscendString &d) const {
  if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  } else {
    return (*name_) <= (*(d.name_));
  }
}

bool AscendString::operator>=(const AscendString &d) const {
  if (d.name_ == nullptr) {
    return true;
  } else if (name_ == nullptr) {
    return false;
  } else {
    return (*name_) >= (*(d.name_));
  }
}

bool AscendString::operator!=(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return true;
  } else {
    return (*name_) != (*(d.name_));
  }
}

bool AscendString::operator==(const char_t *const d) const {
  if ((name_ == nullptr) && (d == nullptr)) {
    return true;
  } else if (name_ == nullptr || d == nullptr) {
    return false;
  } else {
    return (strcmp((*name_).c_str(), d) == 0);
  }
}

bool AscendString::operator!=(const char_t *const d) const {
  if ((name_ == nullptr) && (d == nullptr)) {
    return false;
  } else if (name_ == nullptr || d == nullptr) {
    return true;
  } else {
    return (strcmp((*name_).c_str(), d) != 0);
  }
}

size_t AscendString::Find(const AscendString &ascend_string) const {
  if ((name_ == nullptr) || (ascend_string.name_ == nullptr)) {
    return std::string::npos;
  }
  return name_->find(*(ascend_string.name_));
}
}  // namespace ge
