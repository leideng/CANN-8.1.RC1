/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_GRAPH_UTILS_TYPE_UTILS_INNER_H_
#define INC_GRAPH_UTILS_TYPE_UTILS_INNER_H_

#include <string>
#include "graph/def_types.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "register/register_types.h"
#include "external/register/register_fmk_types.h"

namespace ge {
class TypeUtilsInner {
 public:
  static bool IsDataTypeValid(const DataType dt);
  static bool IsFormatValid(const Format format);
  static bool IsDataTypeValid(std::string dt); // for user json input
  static bool IsFormatValid(std::string format); // for user json input
  static bool IsInternalFormat(const Format format);

  static std::string ImplyTypeToSerialString(const domi::ImplyType imply_type);
  static graphStatus SplitFormatFromStr(const std::string &str, std::string &primary_format_str, int32_t &sub_format);
  static Format DomiFormatToFormat(const domi::domiTensorFormat_t domi_format);
  static std::string FmkTypeToSerialString(const domi::FrameworkType fmk_type);
  static bool CheckUint64MulOverflow(const uint64_t a, const uint32_t b);
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_TYPE_UTILS_INNER_H_
