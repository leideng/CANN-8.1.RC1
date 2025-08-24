/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include <string>
#include <iostream>
#include <fstream>
#include "ascend_ir/ascend_ir_core/ascend_ir.h"
#include "inc/external/graph/utils/type_utils.h"
namespace ge {
class DumpAscirGraph {
 public:
  static std::string DumpGraph(AscGraph &graph);
  static void WriteOutToFile(const std::string &filename, AscGraph &graph);
 private:
  static std::stringstream &TilingKeyStr(std::stringstream &ss, AscGraph &graph);
  static std::stringstream &NameStr(std::stringstream &ss, AscGraph &graph);
  static std::stringstream &AllAxisStr(std::stringstream &ss, AscGraph &graph);
  static std::stringstream &AscNodeAttrStr(std::stringstream &ss, AscNodeAttr &attr);
  static std::stringstream &AscTensorAttrStr(std::stringstream &ss, AscTensorAttr *attr);
  static std::stringstream &MemAttrStr(std::stringstream &ss, AscTensorAttr *attr);
  static std::stringstream &MemQueueAttrStr(std::stringstream &ss, AscTensorAttr *attr);
  static std::stringstream &MemBufAttrStr(std::stringstream &ss, AscTensorAttr *attr);
  static std::stringstream &MemOptAttrStr(std::stringstream &ss, AscTensorAttr *attr);
  static std::stringstream &NodesStr(std::stringstream &ss, ge::AscNodeVisitor &nodes);
  static std::string ApiTypeToString(ge::ApiType type);
  static std::string ComputUnitToString(ge::ComputeUnit unit);
  static std::string ComputeTypeToString(ge::ComputeType type);
  static std::string AllocTypeToString(ge::AllocType type);
  static std::string PositionToString(ge::Position position);
  static std::string HardwareToString(ge::MemHardware hardware);
};
} // namespace ge