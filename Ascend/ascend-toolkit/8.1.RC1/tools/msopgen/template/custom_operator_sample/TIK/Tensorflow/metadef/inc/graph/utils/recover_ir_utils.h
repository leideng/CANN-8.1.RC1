/* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/
#ifndef METADEF_CXX_INC_GRAPH_UTILS_RECOVER_IR_UTILS_H_
#define METADEF_CXX_INC_GRAPH_UTILS_RECOVER_IR_UTILS_H_
namespace ge {
class RecoverIrUtils {
 public:
  using InputIrDefs = std::vector<std::pair<std::string, ge::IrInputType>>;
  using OutputIrDefs = std::vector<std::pair<std::string, ge::IrOutputType>>;
  template<typename IrType>
  using IrDefAppender =
  std::function<void(const ge::OpDescPtr &op_desc, const std::string &ir_name, const IrType ir_type)>;

  struct IrDefinition {
    bool inited{false};
    bool has_ir_definition{false};
    std::vector<std::string> attr_names;
    std::map<std::string, ge::AnyValue> attr_value;
    InputIrDefs inputs;
    OutputIrDefs outputs;
    ge::OpDescPtr op_desc{nullptr};
  };
  static ge::graphStatus RecoverOpDescIrDefinition(const ge::OpDescPtr &desc,
                                                   const std::string &op_type,
                                                   IrDefinition &ir_def);
  static void InitIrDefinitionsIfNeed(const std::string &op_type, IrDefinition &ir_def);
  static graphStatus RecoverIrAttrNames(const ge::OpDescPtr &desc, IrDefinition &ir_def);
  static graphStatus RecoverIrInputAndOutput(const ge::OpDescPtr &desc, IrDefinition &ir_def);
  static graphStatus RecoverIrDefinitions(const ge::ComputeGraphPtr &graph, const vector<std::string> &attr_names = {});
  static graphStatus RecoverOpDescIrDefinition(const ge::OpDescPtr &desc, const std::string &op_type = "");
};
}

#endif // METADEF_CXX_INC_GRAPH_UTILS_RECOVER_IR_UTILS_H_
