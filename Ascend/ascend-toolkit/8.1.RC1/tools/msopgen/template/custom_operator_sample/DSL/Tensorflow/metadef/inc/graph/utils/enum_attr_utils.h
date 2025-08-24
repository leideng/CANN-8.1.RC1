/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef __INC_METADEF_ENUM_ATTR_UTILS_H
#define __INC_METADEF_ENUM_ATTR_UTILS_H

#include <vector>
#include "common/ge_common/util.h"
#include "graph/ge_error_codes.h"
#include "graph/ge_tensor.h"

namespace ge {
using namespace std;
constexpr uint16_t kMaxValueOfEachDigit = 127U;
constexpr size_t kAppendNum = 1U;
constexpr char_t prefix = '\0';

class EnumAttrUtils {
 public:
  static void GetEnumAttrName(vector<string> &enum_attr_names, const string &attr_name, string &enum_attr_name,
                              bool &is_new_attr);
  static void GetEnumAttrValue(vector<string> &enum_attr_values, const string &attr_value, int64_t &enum_attr_value);
  static void GetEnumAttrValues(vector<string> &enum_attr_values, const vector<string> &attr_values,
                                vector<int64_t> &enum_values);

  static graphStatus GetAttrName(const vector<string> &enum_attr_names, const vector<bool> name_use_string_values,
                                 const string &enum_attr_name, string &attr_name, bool &is_value_string);
  static graphStatus GetAttrValue(const vector<string> &enum_attr_values, const int64_t enum_attr_value,
                                  string &attr_value);
  static graphStatus GetAttrValues(const vector<string> &enum_attr_values, const vector<int64_t> &enum_values,
                                   vector<string> &attr_values);
 private:
  static void Encode(const uint32_t src, string &dst);
  static void Decode(const string &src, size_t &dst);
};
} // namespace ge
#endif // __INC_METADEF_ENUM_ATTR_UTILS_H
