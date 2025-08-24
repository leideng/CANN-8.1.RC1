/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
* ===================================================================================================================*/

#include "graph/utils/type_utils.h"

namespace ge {

ge::AscendString TypeUtils::DataTypeToAscendString(const DataType &data_type) {
  return AscendString(DataTypeToSerialString(data_type).c_str());
}

DataType TypeUtils::AscendStringToDataType(const ge::AscendString &str) {
  return SerialStringToDataType(str.GetString());
}

AscendString TypeUtils::FormatToAscendString(const Format &format) {
  return AscendString(FormatToSerialString(format).c_str());
}

Format TypeUtils::AscendStringToFormat(const AscendString &str) {
  return SerialStringToFormat(str.GetString());
}

Format TypeUtils::DataFormatToFormat(const AscendString &str) {
  return DataFormatToFormat(std::string(str.GetString()));
}

}