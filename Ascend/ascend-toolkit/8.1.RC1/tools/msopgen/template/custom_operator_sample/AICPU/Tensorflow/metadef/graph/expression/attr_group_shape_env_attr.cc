/* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "attribute_group/attr_group_shape_env.h"
#include "graph/debug/ge_util.h"
#include "proto/ge_ir.pb.h"
#include "graph/expression/expression_impl.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
static thread_local ShapeEnvAttr *shape_env_context{nullptr};
static std::map<ge::DataType, std::string> kGeDType2CppDtype = {
  {ge::DT_INT32, "int32_t"},
  {ge::DT_INT64, "int64_t"},
  {ge::DT_UINT32, "uint32_t"},
  {ge::DT_UINT64, "uint64_t"},
};
}

ShapeEnvAttr *GetCurShapeEnvContext() {
  return shape_env_context;
}

void SetCurShapeEnvContext(ShapeEnvAttr *shape_env) {
  shape_env_context = shape_env;
}

std::string InputSource::GetSourceStr() const {
  return "GetDimValueFromGraphInputData(" + std::to_string(input_data_idx_) + ", " + std::to_string(dim_idx_) + ");";
}

std::string GraphInputShapeSource::GetSourceStr() const {
  return "GetGraphInputTensor(" + std::to_string(input_data_idx_) + ")->GetOriginShape().GetDim(" +
      std::to_string(dim_idx_) + ")";
}

std::string GraphInputValueSource::GetSourceStr() const {
  if (kGeDType2CppDtype.find(dtype_) == kGeDType2CppDtype.end()) {
    GELOGE(ge::PARAM_INVALID, "Do not support dtype %s while get symbol source",
           TypeUtils::DataTypeToSerialString(dtype_).c_str());
    return "";
  }
  return "GetGraphInputTensor(" + std::to_string(input_data_idx_) + ")->GetData<" + kGeDType2CppDtype[dtype_] +
      ">()[" + std::to_string(dim_idx_) + "];";
}

graphStatus ShapeEnvAttr::SerializeSymbolInfo(proto::ShapeEnvAttrGroupsDef *shape_env_group) {
  GE_ASSERT_NOTNULL(shape_env_group);
  shape_env_group->clear_symbol_to_value();
  auto symbol_to_value_def = shape_env_group->mutable_symbol_to_value();
  GE_ASSERT_NOTNULL(symbol_to_value_def);
  GELOGI("symbol_to_value size: %zu", symbol_to_value_.size());
  for (const auto &iter : symbol_to_value_) {
    GE_ASSERT_TRUE(!iter.first.IsConstExpr(),
        "Symbol in symbol_to_value of shape env attr should be a variable, but get: %s",
        iter.first.Serialize().get());
    symbol_to_value_def->insert({iter.first.Serialize().get(), iter.second});
  }
  auto value_to_symbol_def = shape_env_group->mutable_value_to_symbol();
  GE_ASSERT_NOTNULL(value_to_symbol_def);
  for (const auto &iter : value_to_symbol_) {
    GE_ASSERT_TRUE(!iter.second.empty());
    proto::SymbolInfoDef symbol_infos;
    for (const auto &sym_iter : iter.second) {
      GE_ASSERT_TRUE(!sym_iter.IsConstExpr(),
          "Symbol in value_to_symbol of shape env attr should be a variable, but get: %s",
          sym_iter.Serialize().get());
      symbol_infos.add_symbols(sym_iter.Serialize().get());
    }
    value_to_symbol_def->insert({iter.first, symbol_infos});
  }

  auto symbol_to_source_def = shape_env_group->mutable_symbol_to_source();
  GE_ASSERT_NOTNULL(symbol_to_source_def);
  // todoo: symbol_to_source_实现序列化
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::SerializeSymbolCheckInfos(proto::ShapeEnvAttrGroupsDef *shape_env_group) {
  GE_ASSERT_NOTNULL(shape_env_group);
  auto replacements_def = shape_env_group->mutable_replacements();
  for (const auto &iter : replacements_) {
    proto::ReplacementDef rep_def;
    rep_def.set_replace_expr(iter.second.replace_expr.Serialize().get());
    rep_def.set_rank(iter.second.rank);
    replacements_def->insert({iter.first.Serialize().get(), rep_def});
  }
  shape_env_group->clear_symbol_check_infos();
  for (const auto &iter : symbol_check_infos_) {
    proto::SymbolCheckInfoDef *symbol_check_info_def = shape_env_group->add_symbol_check_infos();
    symbol_check_info_def->set_expr(iter.expr.Serialize().get());
    symbol_check_info_def->set_file(iter.file);
    symbol_check_info_def->set_line(iter.line);
  }
  shape_env_group->clear_symbol_assert_infos();
  for (const auto &iter : symbol_assert_infos_) {
    proto::SymbolCheckInfoDef *symbol_assert_info_def = shape_env_group->add_symbol_assert_infos();
    symbol_assert_info_def->set_expr(iter.expr.Serialize().get());
    symbol_assert_info_def->set_file(iter.file);
    symbol_assert_info_def->set_line(iter.line);
  }
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::Serialize(proto::AttrGroupDef &attr_group_def) {
  auto shape_env_group = attr_group_def.mutable_shape_env_attr_group();
  GE_ASSERT_SUCCESS(SerializeSymbolInfo(shape_env_group));
  GE_ASSERT_SUCCESS(SerializeSymbolCheckInfos(shape_env_group));
  proto::ShapeEnvSettingDef *shape_env_setting_def = shape_env_group->mutable_shape_setting();
  shape_env_setting_def->set_specialize_zero_one(shape_env_setting_.specialize_zero_one);
  shape_env_setting_def->set_dynamic_mode(static_cast<int32_t>(shape_env_setting_.dynamic_mode));
  shape_env_group->set_unique_sym_id(unique_sym_id_);
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::DeserializeSymbolInfo(const proto::ShapeEnvAttrGroupsDef &shape_env_group) {
  symbol_to_value_.clear();
  GELOGI("symbol_to_value size: %zu", shape_env_group.symbol_to_value_size());
  for (const auto &iter : shape_env_group.symbol_to_value()) {
    Expression sym = Expression::Deserialize(iter.first.c_str());
    GE_ASSERT_TRUE(!sym.IsConstExpr(),
        "Symbol in symbol_to_value of shape env attr should be a variable, but get: %s",
        iter.first.c_str());
    symbol_to_value_.emplace(std::make_pair(sym, iter.second));
  }
  value_to_symbol_.clear();
  for (const auto &iter : shape_env_group.value_to_symbol()) {
    std::vector<Expression> symbol_infos;
    for (const auto &sym_iter : iter.second.symbols()) {
      Expression sym = Expression::Deserialize(sym_iter.c_str());
      GE_ASSERT_TRUE(!sym.IsConstExpr(),
          "Symbol in value_to_symbol of shape env attr should be a variable, but get: %s",
          sym_iter.c_str());
      symbol_infos.emplace_back(sym);
    }
    value_to_symbol_.emplace(std::make_pair(iter.first, symbol_infos));
  }
  symbol_to_source_.clear();
  // todoo: symbol_to_source_实现反序列化
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::DeserializeSymbolCheckInfos(const proto::ShapeEnvAttrGroupsDef &shape_env_group) {
  replacements_.clear();
  for (const auto &iter : shape_env_group.replacements()) {
    Expression expr = Expression::Deserialize(iter.first.c_str());
    Expression replace_expr = Expression::Deserialize(iter.second.replace_expr().c_str());
    replacements_.emplace(std::make_pair(expr, Replacement(replace_expr, iter.second.rank())));
  }
  symbol_check_infos_.clear();
  for (const auto &iter : shape_env_group.symbol_check_infos()) {
    Expression expr = Expression::Deserialize(iter.expr().c_str());
    symbol_check_infos_.emplace(SymbolCheckInfo(expr, iter.file(), iter.line()));
  }
  symbol_assert_infos_.clear();
  for (const auto &iter : shape_env_group.symbol_assert_infos()) {
    Expression expr = Expression::Deserialize(iter.expr().c_str());
    symbol_assert_infos_.emplace(SymbolCheckInfo(expr, iter.file(), iter.line()));
  }
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::Deserialize(const proto::AttrGroupDef &attr_group_def) {
  const auto& shape_env_group = attr_group_def.shape_env_attr_group();
  DeserializeSymbolInfo(shape_env_group);
  DeserializeSymbolCheckInfos(shape_env_group);
  shape_env_setting_ =
      ShapeEnvSetting(shape_env_group.shape_setting().specialize_zero_one(),
          static_cast<DynamicMode>(shape_env_group.shape_setting().dynamic_mode()));
  unique_sym_id_ = shape_env_group.unique_sym_id();
  return GRAPH_SUCCESS;
}

std::unique_ptr<AttrGroupsBase> ShapeEnvAttr::Clone() {
  std::unique_ptr<AttrGroupsBase> new_attr = ComGraphMakeUnique<ShapeEnvAttr>(*this);
  GE_ASSERT_NOTNULL(new_attr);
  return new_attr;
}

bool ShapeEnvAttr::HasSymbolCheckInfo(const ge::Expression &expr) const {
  if (symbol_check_infos_.find(SymbolCheckInfo(expr)) != symbol_check_infos_.end()) {
    return true;
  }
  return false;
}

bool ShapeEnvAttr::HasSymbolAssertInfo(const ge::Expression &expr) const {
  if (symbol_assert_infos_.find(SymbolCheckInfo(expr)) != symbol_assert_infos_.end()) {
    return true;
  }
  return false;
}

Symbol ShapeEnvAttr::CreateSymbol(const int64_t hint, const InputSource &source) {
  auto source_ptr = ComGraphMakeShared<GraphInputShapeSource>(source.GetInputDataIdx(), source.GetDimIdx());
  return CreateSymbol(hint, source_ptr);
}

ge::Expression ShapeEnvAttr::FindReplacements(const ge::Expression &expr) {
  auto iter = replacements_.find(expr);
  if (iter == replacements_.end()) {
    return expr;
  }
  if (iter->second.has_replace) {
    GELOGD("Find replace expr: %s of expr: %s has replace",
        iter->second.replace_expr.Str().get(), expr.Str().get());
    return expr;
  }
  auto replace_expr = iter->second.replace_expr;
  GELOGD("Find replace expr: %s of expr: %s",
      replace_expr.Str().get(), expr.Str().get());
  if (replace_expr == expr) {
    return expr;
  }
  std::vector<std::pair<Expression, Expression>> var_replacements;
  iter->second.has_replace = true;
  for (auto &sym : replace_expr.FreeSymbols()) {
    auto replace_sym = FindReplacements(sym);
    var_replacements.emplace_back(std::make_pair(sym, replace_sym));
  }
  iter->second.has_replace = false;
  return replace_expr.Replace(var_replacements);
}

const std::vector<SymbolCheckInfo> ShapeEnvAttr::GetAllSymbolCheckInfos() const {
  std::vector<SymbolCheckInfo> results;
  for (const auto &iter : symbol_check_infos_) {
    results.emplace_back(iter);
  }
  return results;
}

const std::vector<SymbolCheckInfo> ShapeEnvAttr::GetAllSymbolAssertInfos() const {
  std::vector<SymbolCheckInfo> results;
  for (const auto &iter : symbol_assert_infos_) {
    results.emplace_back(iter);
  }
  return results;
};

ge::Expression ShapeEnvAttr::Simplify(const ge::Expression &expr) {
  std::vector<std::pair<Expression, Expression>> var_replacements;
  GELOGD("Simplify origin expr: %s", expr.Str().get());
  // 初始化replacements遍历状态
  for (auto &iter : replacements_) {
    iter.second.has_replace = false;
    GELOGD("Replacement symbol: %s, replacement expr: %s, rank: %d, has_replace: %d",
        iter.first.Str().get(), iter.second.replace_expr.Str().get(),
        iter.second.rank, iter.second.has_replace);
  }
  for (const auto &sym : expr.FreeSymbols()) {
    auto replace_expr = FindReplacements(sym);
    if ((!replace_expr.IsVariableExpr()) || (replace_expr != sym)) {
      var_replacements.emplace_back(std::make_pair(sym, replace_expr));
    }
  }
  if (!var_replacements.empty()) {
    auto result_expr = expr.Replace(var_replacements);
    GELOGI("Simplify expr: %s to expr: %s",
        expr.Serialize().get(), result_expr.Serialize().get());
    GE_ASSERT_NOTNULL(result_expr.impl_);
    return {result_expr.impl_->Simplify()};
  }
  return expr;
}

ge::Expression ShapeEnvAttr::EvaluateExpr(const ge::Expression &expr) {
  std::vector<std::pair<Expression, Expression>> var_to_val;
  auto free_symbols = expr.FreeSymbols();
  for (const auto &free_sym : free_symbols) {
    const auto &iter = symbol_to_value_.find(free_sym);
    if (iter != symbol_to_value_.end()) {
      var_to_val.emplace_back(std::make_pair(free_sym, Symbol(iter->second)));
    }
  }
  return expr.Subs(var_to_val);
}

void ShapeEnvAttr::AppendInitReplacement(const ge::Expression &expr) {
  if (replacements_.find(expr) == replacements_.end()) {
    (void)replacements_.emplace(std::make_pair(expr, Replacement(expr, 1)));
  }
}

graphStatus ShapeEnvAttr::FindRootExpr(const ge::Expression &expr, ge::Expression &root_expr) {
  const auto &iter = replacements_.find(expr);
  GE_ASSERT_TRUE(iter != replacements_.end(), "Can not find replacement of expr: %s", expr.Serialize().get());
  if (iter->second.replace_expr == expr) {
    root_expr = expr;
    return GRAPH_SUCCESS;
  }
  GE_ASSERT_SUCCESS(FindRootExpr(iter->second.replace_expr, root_expr));
  return GRAPH_SUCCESS;
}

std::vector<std::pair<Expression, InputSource>> ShapeEnvAttr::GetAllSymbolToSourceRelation() {
  std::vector<std::pair<Expression, InputSource>> result;
  for (const auto &iter : symbol_to_source_) {
    InputSource source(iter.second->GetInputDataIdx(), iter.second->GetDimIdx());
    result.emplace_back(iter.first, source);
  }
  return result;
}

std::vector<std::pair<Expression, SourcePtr>> ShapeEnvAttr::GetAllSym2Src() {
  std::vector<std::pair<Expression, SourcePtr>> result;
  for (const auto &iter : symbol_to_source_) {
    result.emplace_back(iter.first, iter.second);
  }
  return result;
}

bool Replacement::operator<=(const Replacement &other) {
  // 并查集的根节点优先级： 常数 > 表达式 > 变量
  if (replace_expr.IsConstExpr()) {
    if (other.replace_expr.IsConstExpr()) {
      return rank <= other.rank;
    }
    return false;
  }
  if (replace_expr.IsVariableExpr()) {
    if (other.replace_expr.IsVariableExpr()) {
      return rank <= other.rank;
    }
    return true;
  }
  if (other.replace_expr.IsConstExpr()) {
    return true;
  }
  if (other.replace_expr.IsVariableExpr()) {
    return false;
  }
  return rank <= other.rank;
}

graphStatus ShapeEnvAttr::MergeReplacement(const ge::Expression &expr1,
    const ge::Expression &expr2) {
  ge::Expression father_expr1;
  GE_ASSERT_SUCCESS(FindRootExpr(expr1, father_expr1));
  ge::Expression father_expr2;
  GE_ASSERT_SUCCESS(FindRootExpr(expr2, father_expr2));
  auto &replacement_1 = replacements_[father_expr1];
  auto &replacement_2 = replacements_[father_expr2];
  if (replacement_1 <= replacement_2) {
    replacement_1.replace_expr = father_expr2;
    if (replacement_2.rank <= replacement_1.rank) {
      replacement_2.rank = replacement_1.rank + 1;
    }
  } else {
    replacement_2.replace_expr = father_expr1;
    if (replacement_1.rank <= replacement_2.rank) {
      replacement_1.rank = replacement_2.rank + 1;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::MergePath() {
  for (auto &iter : replacements_) {
    ge::Expression root_expr;
    GE_ASSERT_SUCCESS(FindRootExpr(iter.first, root_expr));
    iter.second.replace_expr = root_expr;
    iter.second.rank = 1;
  }
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::AppendReplacement(const ge::Expression &expr1,
    const ge::Expression &expr2) {
  if (expr1 == expr2) {
    return GRAPH_SUCCESS;
  }
  // 仅支持 符号->常量，符号->表达式，符号->符号 映射
  if (expr1.IsConstExpr()) {
    if (!expr2.IsVariableExpr()) {
      GELOGW("Unsupport append replacement %s to %s",
          expr1.Serialize().get(), expr2.Serialize().get());
      return GRAPH_SUCCESS;
    }
  } else if (!expr1.IsVariableExpr()) {
    if (!expr2.IsVariableExpr()) {
      GELOGW("Unsupport append replacement %s to %s",
          expr1.Serialize().get(), expr2.Serialize().get());
      return GRAPH_SUCCESS;
    }
  }
  AppendInitReplacement(expr1);
  AppendInitReplacement(expr2);
  GE_ASSERT_SUCCESS(MergeReplacement(expr1, expr2));
  // 路径压缩
  GE_ASSERT_SUCCESS(MergePath());
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::AppendSymbolAssertInfo(const ge::Expression &expr,
    const std::string &file, const int64_t line) {
  GE_ASSERT_TRUE(expr.IsBooleanExpr(),
      "Assert expr: %s should be boolean", expr.Serialize().get());
  if (!expr.IsConstExpr()) {
    (void)symbol_assert_infos_.emplace(SymbolCheckInfo(expr, file, line));
  }
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::AppendSymbolCheckInfo(const ge::Expression &expr,
    const std::string &file, const int64_t line) {
  GE_ASSERT_TRUE(expr.IsBooleanExpr(),
      "Check expr: %s should be boolean", expr.Serialize().get());
  if (!expr.IsConstExpr()) {
    (void)symbol_check_infos_.emplace(SymbolCheckInfo(expr, file, line));
  }
  return GRAPH_SUCCESS;
}
} // namespace ge
