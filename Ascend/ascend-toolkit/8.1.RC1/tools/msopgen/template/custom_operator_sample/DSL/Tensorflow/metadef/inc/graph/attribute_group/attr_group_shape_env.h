/* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
* ===================================================================================================================*/

#ifndef INC_GRAPH_ATTR_GROUP_ATTR_GROUP_SHAPE_ENV_H
#define INC_GRAPH_ATTR_GROUP_ATTR_GROUP_SHAPE_ENV_H

#include <unordered_map>
#include <unordered_set>
#include "attr_group_base.h"
#include "graph/symbolic.h"
#include "common/checker.h"

namespace ge {
namespace proto {
class AttrGroupDef;
class ShapeEnvAttrGroupsDef;
}

ShapeEnvAttr *GetCurShapeEnvContext();
void SetCurShapeEnvContext(ShapeEnvAttr *shape_env);

class Source {
 public:
  virtual ~Source() = default;
  virtual std::string GetSourceStr() const = 0;
  // todoo: 兼容上库，待删除
  virtual int32_t GetInputDataIdx() const {
    return std::numeric_limits<int32_t>::max();
  };
  virtual size_t GetDimIdx() const {
    return std::numeric_limits<size_t>::max();
  }
};
using SourcePtr = std::shared_ptr<Source>;

class GraphInputShapeSource : public Source {
 public:
  GraphInputShapeSource(int32_t input_data_idx, size_t dim_idx) : input_data_idx_(input_data_idx), dim_idx_(dim_idx) {}

  int32_t GetInputDataIdx() const override {
    return input_data_idx_;
  }

  size_t GetDimIdx() const override {
    return dim_idx_;
  }

  std::string GetSourceStr() const override;
  bool operator==(const GraphInputShapeSource &other) const {
    return (this->input_data_idx_ == other.input_data_idx_) && (this->dim_idx_ == other.dim_idx_);
  }

 private:
  int32_t input_data_idx_;  // Data的index，描述symbol来自于graph输入中第几个输入data
  size_t dim_idx_;          // 描述symbol来自于data中对应shape的第几个dim
};

class GraphInputValueSource : public Source {
 public:
  GraphInputValueSource(int32_t input_data_idx, size_t dim_idx, ge::DataType dtype)
      : input_data_idx_(input_data_idx), dim_idx_(dim_idx), dtype_(dtype) {}

  std::string GetSourceStr() const override;
  bool operator==(const GraphInputValueSource &other) const {
    return (this->input_data_idx_ == other.input_data_idx_) && (this->dim_idx_ == other.dim_idx_) &&
        (this->dtype_ == other.dtype_);
  }

 private:
  int32_t input_data_idx_;  // Data的index，描述symbol来自于graph输入中第几个输入data
  size_t dim_idx_;          // 描述symbol来自于data中对应shape的第几个dim
  ge::DataType dtype_;      // 描述value的数据类型，用于后续执行时取值
};

class InputSource {
 public:
  InputSource(int32_t input_data_idx, size_t dim_idx) : input_data_idx_(input_data_idx), dim_idx_(dim_idx) {}

  int32_t GetInputDataIdx() const {
    return input_data_idx_;
  }

  size_t GetDimIdx() const { return dim_idx_; }

  std::string GetSourceStr() const;
  bool operator==(const InputSource &other) const {
    return (this->input_data_idx_ == other.input_data_idx_) && (this->dim_idx_ == other.dim_idx_);
  }
  InputSource() = default;
 private:
  int32_t input_data_idx_{-1}; // Data的index，描述symbol来自于graph输入中第几个输入data
  size_t dim_idx_{0}; // 描述symbol来自于data中对应shape的第几个dim
};

struct HashSymbol {
  size_t operator()(const Expression &e) const {
    int64_t value_int = 0L;
    double value_float = 0.0f;
    bool value_bool = false;
    switch (e.GetExprType()) {
      case ExprType::kExprConstantBoolean:
        GE_ASSERT_TRUE(e.GetConstValue(value_bool));
        return std::hash<bool>()(value_bool);
      case ExprType::kExprConstantInteger:
        GE_ASSERT_TRUE(e.GetConstValue(value_int));
        return std::hash<int64_t>()(value_int);
      case ExprType::kExprConstantRealDouble:
      case ExprType::kExprConstantRation:
        GE_ASSERT_TRUE(e.GetConstValue(value_float));
        return std::hash<double>()(value_float);
      default:
        return std::hash<std::string>()(std::string(e.Serialize().get()));
    }
  }
};

struct SymbolCheckInfo {
  ge::Expression expr;
  std::string file;
  int64_t line{};
  explicit SymbolCheckInfo(const ge::Expression &in_expr,
      const std::string &in_file = "", const int64_t in_line = -1)
       : expr(in_expr), file(in_file), line(in_line) {}
  SymbolCheckInfo() = default;
  bool operator==(const SymbolCheckInfo &other) const {
    return this->expr == other.expr;
  }
};

struct HashSymbolCheckInfo {
  size_t operator()(const SymbolCheckInfo &s) const {
    if (s.expr.GetExprType() == ExprType::kExprConstantBoolean) {
      bool value = false;
      GE_ASSERT_TRUE(s.expr.GetConstValue(value));
      return std::hash<bool>()(value);
    }
    return std::hash<std::string>()(std::string(s.expr.Serialize().get()));
  }
};

// 配置符号的生成方式
// dynamic：不管hint值是否相等，均生成新符号
// duck：当hint值相同时，则不生成新符号，使用之前生成过的符号
// static：根据hint值生成符号，同时添加一个Assert（sym == hint）的guard
enum class DynamicMode {
  kDynamic = 0,
  kDuck = 1,
  kStatic = 2,
  kEnd = 3
};

struct ShapeEnvSetting {
  bool specialize_zero_one{false};
  DynamicMode dynamic_mode{DynamicMode::kDynamic};
  ShapeEnvSetting() = default;
  ShapeEnvSetting(const bool in_specialize_zero_one, const DynamicMode &in_dynamic_mode)
      : specialize_zero_one(in_specialize_zero_one), dynamic_mode(in_dynamic_mode) {}
};

struct Replacement {
  ge::Expression replace_expr;
  int32_t rank;
  bool has_replace;
  Replacement(const ge::Expression &a, const int32_t in_rank, bool in_has_replace = false)
       : replace_expr(a), rank(in_rank), has_replace(in_has_replace) {}
  Replacement() : rank(0), has_replace(false) {}
  bool operator<=(const Replacement &other);
};

class ShapeEnvAttr : public AttrGroupsBase {
 public:
  ShapeEnvAttr() = default;
  ~ShapeEnvAttr() override = default;
  explicit ShapeEnvAttr(const ShapeEnvSetting &shape_env_setting) : shape_env_setting_(shape_env_setting) {}
  graphStatus Serialize(proto::AttrGroupDef &attr_group_def) override;
  graphStatus Deserialize(const proto::AttrGroupDef &attr_group_def) override;

  // todoo：兼容模式上库，待删除该接口，使用CreateSymbol<T>\GetAllSym2Src接口代替
  Symbol CreateSymbol(const int64_t hint, const InputSource &source);
  std::vector<std::pair<Expression, InputSource>> GetAllSymbolToSourceRelation();

  // 只支持int32，uint32, int64, uint64
  template<typename T>
  typename std::enable_if<std::is_integral<T>::value, Symbol>::type CreateSymbol(T hint, const SourcePtr &source) {
    auto hint_int64 = static_cast<int64_t>(hint);
    GE_ASSERT_TRUE((shape_env_setting_.dynamic_mode >= DynamicMode::kDynamic) &&
                       (shape_env_setting_.dynamic_mode < DynamicMode::kEnd),
                   "Invalid dynamic mode: %d, create symbol failed", shape_env_setting_.dynamic_mode);
    if (shape_env_setting_.specialize_zero_one && ((hint_int64 == 0) || (hint_int64 == 1))) {
      GELOGI("Create symbol %d for in specialize_zero_one mode, source: %s", hint_int64,
             source->GetSourceStr().c_str());
      return Symbol(hint_int64);
    }
    if (shape_env_setting_.dynamic_mode != DynamicMode::kDynamic) {
      // 非动态模式，hint值相同使用同一个符号
      const auto &iter = value_to_symbol_.find(hint_int64);
      if (iter != value_to_symbol_.end()) {
        GE_ASSERT_TRUE(!iter->second.empty());
        return Symbol(iter->second.front().Serialize().get());
      }
    }
    GE_ASSERT_TRUE(unique_sym_id_ + 1 < std::numeric_limits<uint64_t>::max(),
                   "unique_sym_id_ is " PRIu64 ". will reach the maximum value of uint64.", unique_sym_id_);
    const std::string sym_name = "s" + std::to_string(unique_sym_id_++);
    auto sym = Symbol(sym_name.c_str());
    symbol_to_source_.emplace(sym, source);
    symbol_to_value_.emplace(sym, hint_int64);
    const auto iter = value_to_symbol_.find(hint_int64);
    if (iter != value_to_symbol_.end()) {
      iter->second.emplace_back(sym);
    } else {
      std::vector<Expression> syms = {sym};
      value_to_symbol_.emplace(hint_int64, syms);
    }
    // 静态场景需要增加一个s == hint的Assert信息
    if (shape_env_setting_.dynamic_mode == DynamicMode::kStatic) {
      ASSERT_SYMBOL_EQ(sym, Symbol(hint_int64));
    }
    return sym;
  }
  std::vector<std::pair<Expression, SourcePtr>> GetAllSym2Src();

  ge::Expression Simplify(const ge::Expression &expr);
  ge::Expression EvaluateExpr(const ge::Expression &expr);
  graphStatus AppendReplacement(const ge::Expression &expr1, const ge::Expression &expr2);
  graphStatus AppendSymbolAssertInfo(const ge::Expression &expr,
      const std::string &file = "", const int64_t line = 0L);
  graphStatus AppendSymbolCheckInfo(const ge::Expression &expr,
      const std::string &file = "", const int64_t line = 0L);
  const std::vector<SymbolCheckInfo> GetAllSymbolCheckInfos() const;
  const std::vector<SymbolCheckInfo> GetAllSymbolAssertInfos() const;
  bool HasSymbolCheckInfo(const ge::Expression &expr) const;
  bool HasSymbolAssertInfo(const ge::Expression &expr) const;
  std::unique_ptr<AttrGroupsBase> Clone() override;
 private:
  void AppendInitReplacement(const ge::Expression &expr);
  ge::Expression FindReplacements(const ge::Expression &expr);
  graphStatus MergeReplacement(const ge::Expression &expr1, const ge::Expression &expr2);
  graphStatus FindRootExpr(const ge::Expression &expr, ge::Expression &root_expr);
  graphStatus SerializeSymbolCheckInfos(proto::ShapeEnvAttrGroupsDef *shape_env_group);
  graphStatus MergePath();
  graphStatus SerializeSymbolInfo(proto::ShapeEnvAttrGroupsDef *shape_env_group);
  graphStatus DeserializeSymbolInfo(const proto::ShapeEnvAttrGroupsDef &shape_env_group);
  graphStatus DeserializeSymbolCheckInfos(const proto::ShapeEnvAttrGroupsDef &shape_env_group);
  std::unordered_map<ge::Expression, ge::Replacement, HashSymbol> replacements_;
  std::unordered_map<Expression, int64_t, HashSymbol> symbol_to_value_;
  std::unordered_map<int64_t, std::vector<Expression>> value_to_symbol_;
  std::unordered_map<Expression, SourcePtr, HashSymbol> symbol_to_source_;
  std::unordered_set<SymbolCheckInfo, HashSymbolCheckInfo> symbol_check_infos_;
  std::unordered_set<SymbolCheckInfo, HashSymbolCheckInfo> symbol_assert_infos_;
  ShapeEnvSetting shape_env_setting_;
  uint64_t unique_sym_id_{0U};
};

}

#endif  // INC_GRAPH_ATTR_GROUP_ATTR_GROUP_SHAPE_ENV_H
