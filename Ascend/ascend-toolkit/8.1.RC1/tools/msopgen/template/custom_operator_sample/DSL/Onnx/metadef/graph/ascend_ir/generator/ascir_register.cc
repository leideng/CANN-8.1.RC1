/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/
#include "graph/ascend_ir/ascir_register.h"
#include "graph/ascend_ir/ascir_registry.h"
#include "graph/types.h"
namespace ge {
namespace ascir {
AscirRegister::AscirRegister(const char *type, const char *def_file_path, int64_t line)
    : ir_def_{} {
  ir_def_.type = type;
  ir_def_.file_path = def_file_path;
  ir_def_.line = line;
  ir_def_.start_node = false;
}

AscirRegister &AscirRegister::Inputs(std::vector<ge::AscendString> &&input_names) {
  for (const auto &input_name: input_names) {
    ir_def_.input_defs.emplace_back(input_name.GetString(), ge::IrInputType::kIrInputRequired);
  }
  return *this;
}

AscirRegister &AscirRegister::DynamicInput(const std::string &input_name) {
  ir_def_.input_defs.emplace_back(input_name, ge::IrInputType::kIrInputDynamic);
  return *this;
}

AscirRegister &AscirRegister::OptionalInput(const std::string &input_name) {
  ir_def_.input_defs.emplace_back(input_name, ge::IrInputType::kIrInputOptional);
  return *this;
}

AscirRegister &AscirRegister::Outputs(std::vector<ge::AscendString> &&output_names) {
  for (const auto &output_name: output_names) {
    ir_def_.output_defs.emplace_back(output_name.GetString(), ge::IrOutputType::kIrOutputRequired);
  }
  return *this;
}

AscirRegister &AscirRegister::DynamicOutput(const std::string &output_name) {
  ir_def_.output_defs.emplace_back(output_name, ge::IrOutputType::kIrOutputDynamic);
  return *this;
}

AscirRegister::AscirRegister(const AscirRegister &other) {
  AscirRegistry::GetInstance().RegisterAscIr(other.ir_def_.type, other.ir_def_);
}
AscirRegister &AscirRegister::Attr(std::string name, std::string asc_type, std::string ge_type) {
  if (ir_def_.IsAttrExisted(name)) {
    return *this;
  }
  ir_def_.attr_defs.emplace_back(AscIrAttrDef{std::move(name), std::move(asc_type), std::move(ge_type)});
  return *this;
}
AscirRegister &AscirRegister::StartNode() {
  ir_def_.start_node = true;
  return *this;
}
AscirRegister &AscirRegister::InferDataType(AscIrDef::CodeGenerator infer_data_type_generator) {
  ir_def_.infer_data_type_generator = std::move(infer_data_type_generator);
  return *this;
}
AscirRegister &AscirRegister::InferView(AscIrDef::CodeGenerator infer_view_generator) {
  ir_def_.infer_view_generator = std::move(infer_view_generator);
  return *this;
}

AscirRegister &AscirRegister::Views(const std::vector<ViewPolicy> &views_policy) {
  ir_def_.output_views_policy = views_policy;
  return InferView(InferViewByPolicy);
}
AscirRegister &AscirRegister::DataTypes(const std::vector<DtypePolicy> &data_types_policy) {
  ir_def_.output_dtypes_policy = data_types_policy;
  return InferDataType(InferDtypeByPolicy);
}
AscirRegister &AscirRegister::Input(const char_t *input_name, const char_t *datatype_symbol) {
  ir_def_.input_defs.emplace_back(input_name, ge::IrInputType::kIrInputRequired);
  ir_def_.dtype_symbol_store.SetInputSymbol(input_name, ge::kIrInputRequired, datatype_symbol);
  return *this;
}
AscirRegister &AscirRegister::Output(const char_t *output_name, const char_t *datatype_symbol) {
  ir_def_.output_defs.emplace_back(output_name, ge::IrOutputType::kIrOutputRequired);
  ir_def_.dtype_symbol_store.SetOutputSymbol(output_name, ge::kIrOutputRequired, datatype_symbol);
  return *this;
}
AscirRegister &AscirRegister::DataType(const char_t *datatype_symbol, const TensorType &type_range) {
  ir_def_.dtype_symbol_store.DeclareSymbol(datatype_symbol, type_range);
  return *this;
}

AscirRegister &AscirRegister::DynamicInput(const char_t *input_name, const char_t *datatype_symbol) {
  ir_def_.input_defs.emplace_back(input_name, ge::IrInputType::kIrInputDynamic);
  ir_def_.dtype_symbol_store.SetInputSymbol(input_name, ge::kIrInputDynamic, datatype_symbol);
  return *this;
}

AscirRegister &AscirRegister::DataType(const char_t *datatype_symbol, const OrderedTensorTypeList &type_range) {
  ir_def_.dtype_symbol_store.DeclareSymbol(datatype_symbol, type_range);
  return *this;
}

AscirRegister &AscirRegister::CalcTmpBufSize(const std::string calc_tmp_buf_size_func) {
  if (!ir_def_.calc_tmp_buf_size_func.func_name.empty()) {
    GELOGE(ge::FAILED, "has registered calc_tmp_buf_size_func: %s", ir_def_.calc_tmp_buf_size_func.func_name.c_str());
    return *this;
  }
  ir_def_.calc_tmp_buf_size_func = CalcTmpBufSizeFunc{std::move(calc_tmp_buf_size_func), CalcTmpBufSizeFuncType::CustomizeType};
  return *this;
}
AscirRegister &AscirRegister::SameTmpBufSizeFromFirstInput() {
  if (!ir_def_.calc_tmp_buf_size_func.func_name.empty()) {
    GELOGE(ge::FAILED, "has registered calc_tmp_buf_size_func: %s", ir_def_.calc_tmp_buf_size_func.func_name.c_str());
    return *this;
  }
  ir_def_.calc_tmp_buf_size_func = CalcTmpBufSizeFunc{"SameTmpBufSizeWithFirstInput", CalcTmpBufSizeFuncType::CommonType};
  return *this;
}

template<>
AscirRegister &AscirRegister::Attr<float>(ge::AscendString &&name) {
  return Attr(name.GetString(), "float", "Float");
}

template<>
AscirRegister &AscirRegister::Attr<ge::DataType>(ge::AscendString &&name) {
  return Attr(name.GetString(), "ge::DataType", "Int");
}
template<>
AscirRegister &AscirRegister::Attr<ge::Tensor>(ge::AscendString &&name) {
  return Attr(name.GetString(), "ge::Tensor", "Tensor");
}
template<>
AscirRegister &AscirRegister::Attr<std::string>(ge::AscendString &&name) {
  return Attr(name.GetString(), "std::string", "String");
}
template<>
AscirRegister &AscirRegister::Attr<int64_t>(ge::AscendString &&name) {
  return Attr(name.GetString(), "int64_t", "Int");
}
template<>
AscirRegister &AscirRegister::Attr<std::vector<std::vector<int64_t>>>(ge::AscendString &&name) {
  return Attr(name.GetString(), "std::vector<std::vector<int64_t>>", "ListListInt");
}
template<>
AscirRegister &AscirRegister::Attr<ge::Format>(ge::AscendString &&name) {
  return Attr(name.GetString(), "ge::Format", "Int");
}
template<>
AscirRegister &AscirRegister::Attr<ge::Expression>(ge::AscendString &&name) {
  return Attr(name.GetString(), "ge::Expression", "ge::Expression");
}

AscirRegister &AscirRegister::Comment(const string &comment) {
  ir_def_.comment = comment;
  return *this;
}
}  // namespace ascir
}
