/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include <memory>
#include <vector>
#include <map>
#include <symengine/rational.h>
#include "inc/graph/symbolic.h"
#include "attribute_group/attr_group_shape_env.h"
#include "expression_impl.h"
#include "graph/debug/ge_util.h"
#include "graph/utils/math_util.h"
#include "common/checker.h"

namespace ge {
Expression::~Expression() {}

Expression::Expression(Expression &&other) noexcept {
  impl_ = std::move(other.impl_);
}

Expression &Expression::operator=(const Expression &other) {
  if (&other != this) {
    impl_ = ComGraphMakeUnique<ExpressionImpl>();
    if ((other.impl_ != nullptr) && (impl_ != nullptr)) {
      *impl_ = *other.impl_;
    }
  }
  return *this;
}

Expression::Expression(const Expression &other) {
  // Copy
  impl_ = ComGraphMakeUnique<ExpressionImpl>();
  if ((other.impl_ != nullptr) && (impl_ != nullptr)) {
    *impl_ = *other.impl_;
  }
}

Expression &Expression::operator=(Expression &&other) noexcept {
  if (&other != this) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}

std::unique_ptr<char_t[]> Expression::Str(const StrType type) const {
  if (impl_ != nullptr) {
    auto str = impl_->Str(type);
    if (str.empty()) {
      return nullptr;
    }
    auto uni_ptr = ComGraphMakeUnique<char_t[]>(str.size() + 1);
    IF_NULL_RETURN_NULL(uni_ptr);
    // 当src size < dst size时，strncpy_s会在末尾str.size()位置添加'\0'
    GE_ASSERT_EOK(strncpy_s(uni_ptr.get(), str.size() + 1, str.c_str(), str.size()));
    return uni_ptr;
  }
  return nullptr;
}

Expression Expression::Parse(const char_t *str) {
  return {ExpressionImpl::Parse(str)};
}

std::unique_ptr<char_t[]> Expression::Serialize() const {
  return Str(StrType::kStrCpp);
}

std::string Expression::ToString() const {
  auto ret = Str(StrType::kStrCpp);
  return (ret != nullptr) ? ret.get() : "";
}

Expression Expression::Deserialize(const ge::char_t *str) {
  return {ExpressionImpl::Deserialize(str)};
}

ExprType Expression::GetExprType() const {
  if (impl_ != nullptr) {
    return impl_->GetExprType();
  }
  return ExprType::kExprNone;
}

bool Expression::IsConstExpr() const {
  if (impl_!= nullptr) {
    return impl_->IsConstExpr();
  }
  return false;
}

bool Expression::IsVariableExpr() const {
  if (impl_!= nullptr) {
    return impl_->IsVariableExpr();
  }
  return false;
}

bool Expression::IsBooleanExpr() const {
  if (impl_!= nullptr) {
    return impl_->IsBooleanExpr();
  }
  return false;
}

Expression Expression::Replace(const std::vector<std::pair<Expression, Expression>> &replace_vars) const {
  if (impl_ != nullptr) {
    std::map<ExpressionImpl *, ExpressionImpl *> impl_map;
    for (auto &item : replace_vars) {
      impl_map[item.first.impl_.get()] = item.second.impl_.get();
    }
    return {impl_->Replace(impl_map)};
  }
  return {nullptr};
}

Expression Expression::Subs(const std::vector<std::pair<Expression, Expression>> &subs_vars) const {
  if (impl_ != nullptr) {
    std::map<ExpressionImpl *, ExpressionImpl *> impl_map;
    for (auto &item : subs_vars) {
      impl_map[item.first.impl_.get()] = item.second.impl_.get();
    }
    return {impl_->Subs(impl_map)};
  }
  return {nullptr};
}

std::vector<Expression> Expression::FreeSymbols() const {
  if (impl_!= nullptr) {
    std::vector<Expression> ret;
    for (auto &free_symbol : impl_->FreeSymbols()) {
      ret.emplace_back(Expression(std::move(free_symbol)));
    }
    return ret;
  }
  return {};
}

graphStatus Expression::GetResult(const std::vector<std::pair<Expression, Expression>> &vars_value,
                                  double &result) const {
  Expression replace_expr = Replace(vars_value);
  if ((replace_expr.impl_ != nullptr) && (replace_expr.impl_->GetResult(result))) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

bool Expression::IsValid() const {
  return impl_ != nullptr;
}

// 模板函数的定义
template<typename T>
typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
Expression::GetConstValue(T &value) const {
  if (!IsConstExpr() || impl_== nullptr) {
    return false;
  }
  return impl_->GetConstValue(value);
}

// 显式实例化
template bool Expression::GetConstValue<int32_t>(int32_t &) const;    // 实例化 int32 类型
template bool Expression::GetConstValue<uint32_t>(uint32_t &) const;  // 实例化 uint32 类型
template bool Expression::GetConstValue<int64_t>(int64_t &) const;    // 实例化 int64 类型
template bool Expression::GetConstValue<uint64_t>(uint64_t &) const;  // 实例化 uint64 类型
template bool Expression::GetConstValue<double>(double &) const;      // 实例化 double 类型
template bool Expression::GetConstValue<float>(float &) const;        // 实例化 float 类型
template bool Expression::GetConstValue<bool>(bool &) const;          // 实例化 bool 类型

Expression Expression::operator+(const Expression &other) const {
  return sym::Add(*this, other);
}

Expression Expression::operator-(const Expression &other) const {
  return sym::Sub(*this, other);
}

Expression Expression::operator*(const Expression &other) const {
  return sym::Mul(*this, other);
}

Expression Expression::operator/(const Expression &other) const {
  return sym::Div(*this, other);
}

Expression Expression::Simplify() const {
  if (GetCurShapeEnvContext() != nullptr) {
    return GetCurShapeEnvContext()->Simplify(*this);
  }
  if (impl_ != nullptr) {
    return {impl_->Simplify()};
  }
  return {nullptr};
}

bool Expression::ContainVar(const Expression &e) const {
  if (impl_ != nullptr) {
    return impl_->ContainVar(e.impl_.get());
  }
  return false;
}

bool Expression::operator==(const Expression &e) const {
  if (impl_ != nullptr && e.impl_ != nullptr) {
    return (*impl_ == *e.impl_);
  }
  return false;
}

bool Expression::operator!=(const Expression &e) const {
  return !(*this == e);
}

std::ostream &operator<<(std::ostream &os, const Expression &e) {
  if (e.impl_ != nullptr) {
    os << *e.impl_;
  }
  return os;
}

Expression::Expression(ExpressionImplPtr &&e)
    : impl_(std::move(e)) {}

Expression::Expression() {
  impl_ = ge::ComGraphMakeUnique<ExpressionImpl>("");
}

Symbol::Symbol(ExpressionImplPtr &&e) : Expression(std::move(e)) {}

Symbol::Symbol(int32_t value, const char_t *name) {
  impl_ = ge::ComGraphMakeUnique<ExpressionImpl>(value, name);
}

Symbol::Symbol(int64_t value, const char_t *name) {
  impl_ = ge::ComGraphMakeUnique<ExpressionImpl>(value, name);
}
Symbol::Symbol(uint32_t value, const char_t *name) {
  impl_ = ge::ComGraphMakeUnique<ExpressionImpl>(value, name);
}
Symbol::Symbol(uint64_t value, const char_t *name) {
  impl_ = ge::ComGraphMakeUnique<ExpressionImpl>(value, name);
}
Symbol::Symbol(double value, const char_t *name) {
  impl_ = ge::ComGraphMakeUnique<ExpressionImpl>(value, name);
}

Symbol::Symbol(const char_t *name) {
  impl_ = ge::ComGraphMakeUnique<ExpressionImpl>(name);
}

std::unique_ptr<char_t[]> Symbol::GetName() const {
  if (impl_ != nullptr) {
    auto str = impl_->GetName();
    if (str.empty()) {
      return nullptr;
    }
    auto uni_ptr = ComGraphMakeUnique<char_t[]>(str.size() + 1U);
    IF_NULL_RETURN_NULL(uni_ptr);
    // 当src size < dst size时，strncpy_s会在末尾str.size()位置添加'\0'
    GE_ASSERT_EOK(strncpy_s(uni_ptr.get(), str.size() + 1, str.c_str(), str.size()));
    return uni_ptr;
  }
  return nullptr;
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
Expression::ComputeHint(T &hint) const {
  if (IsConstExpr()) {
    return GetConstValue(hint);
  }
  if (GetCurShapeEnvContext() == nullptr) {
    GELOGW("Shape env is nullptr, cannot compute hint, expr: %s", Serialize().get());
    return false;
  }
  return GetCurShapeEnvContext()->EvaluateExpr(*this).GetConstValue(hint);
}

template bool Expression::ComputeHint<int32_t>(int32_t &) const;    // 实例化 int32 类型
template bool Expression::ComputeHint<uint32_t>(uint32_t &) const;  // 实例化 uint32 类型
template bool Expression::ComputeHint<int64_t>(int64_t &) const;    // 实例化 int64 类型
template bool Expression::ComputeHint<uint64_t>(uint64_t &) const;  // 实例化 uint64 类型
template bool Expression::ComputeHint<double>(double &) const;      // 实例化 double 类型
template bool Expression::ComputeHint<float>(float &) const;        // 实例化 float 类型
template bool Expression::ComputeHint<bool>(bool &) const;          // 实例化 bool 类型

Status Expression::AppendSymbolEquivalence(const Expression &e0, const Expression &e1) {
  if (GetCurShapeEnvContext() == nullptr) {
    GELOGW("Shape env is nullptr, cannot append symbol equivalence, expr0: %s, expr1: %s",
        e0.Serialize().get(), e1.Serialize().get());
    return GRAPH_SUCCESS;
  }
  GE_ASSERT_SUCCESS(GetCurShapeEnvContext()->AppendReplacement(e0, e1));
  return GRAPH_SUCCESS;
}

bool Expression::StaticCheckEq(const Expression &e) const {
  return StaticCheckBool(sym::Eq(*this, e));
}

bool Expression::StaticCheckNe(const Expression &e) const {
  return StaticCheckBool(sym::Ne(*this, e));
}

bool Expression::StaticCheckLt(const Expression &e) const {
  return StaticCheckBool(sym::Lt(*this, e));
}

bool Expression::StaticCheckLe(const Expression &e) const {
  return StaticCheckBool(sym::Le(*this, e));
}

bool Expression::StaticCheckGt(const Expression &e) const {
  return StaticCheckBool(sym::Gt(*this, e));
}

bool Expression::StaticCheckGe(const Expression &e) const {
  return StaticCheckBool(sym::Ge(*this, e));
}

bool Expression::StaticCheckBool(const Expression &expr) {
  GE_ASSERT_TRUE(expr.IsBooleanExpr(), "Only boolean expr can do static check, expr: %s",
      expr.Serialize().get());
  bool value = false;
  if (expr.IsConstExpr()) {
    GE_ASSERT_TRUE(expr.GetConstValue(value));
    return value;
  }
  if (GetCurShapeEnvContext() == nullptr) {
    GELOGW("Shape env is nullptr, cannot do static check, expr: %s", expr.Serialize().get());
    return false;
  }
  if ((GetCurShapeEnvContext()->HasSymbolCheckInfo(expr)) || (GetCurShapeEnvContext()->HasSymbolAssertInfo(expr))) {
    return true;
  }
  return false;
}

namespace sym {
Expression operator+(const Expression &e1, const Expression &e2) {
  return Add(e1, e2);
}

Expression operator-(const Expression &e1, const Expression &e2) {
  return Sub(e1, e2);
}

Expression operator*(const Expression &e1, const Expression &e2) {
  return Mul(e1, e2);
}

Expression operator/(const Expression &e1, const Expression &e2) {
  return Div(e1, e2);
}

Expression Add(const Expression &a, const Expression &b) {
  return {Add(a.impl_, b.impl_)};
}

Expression Sub(const Expression &a, const Expression &b) {
  return {Sub(a.impl_, b.impl_)};
}

Expression Mul(const Expression &a, const Expression &b) {
  return {Mul(a.impl_, b.impl_)};
}

Expression Div(const Expression &a, const Expression &b) {
  return {Div(a.impl_, b.impl_)};
}

Expression Max(const Expression &a, const Expression &b) {
  return {Max(a.impl_, b.impl_)};
}

Expression Min(const Expression &a, const Expression &b) {
  return {Min(a.impl_, b.impl_)};
}

Expression Abs(const Expression &a) {
  return {Abs(a.impl_)};
}

Expression Pow(const Expression &a, const Expression &b) {
  return {Pow(a.impl_, b.impl_)};
}

Expression Log(const Expression &a) {
  return {Log(a.impl_)};
}

Expression Log(const Expression &arg, const Expression &base) {
  return {Log(arg.impl_, base.impl_)};
}

Expression Ceiling(const Expression &a) {
  return {Ceiling(a.impl_)};
}

Expression Coeff(const Expression &b, const Expression &x, const Expression &n) {
  return {Coeff(b.impl_, x.impl_, n.impl_)};
}

Expression Rational(int32_t num, int32_t den) {
  auto left = ExpressionImpl::CreateExpressionImpl(num);
  auto right = ExpressionImpl::CreateExpressionImpl(den);
  return {Rational(left, right)};
}

Expression Align(const Expression &arg, uint32_t alignment) {
  auto align = Symbol(alignment, ("alignment_" + std::to_string(alignment)).c_str());
  return Mul(Ceiling(Div(arg, align)), align);
}

Expression Eq(const Expression &a, const Expression &b) {
  return {Eq(a.impl_, b.impl_)};
}

Expression Ne(const Expression &a, const Expression &b) {
  return {Ne(a.impl_, b.impl_)};
}

Expression Ge(const Expression &a, const Expression &b) {
  return {Le(b.impl_, a.impl_)};
}

Expression Gt(const Expression &a, const Expression &b) {
  return {Lt(b.impl_, a.impl_)};
}

Expression Le(const Expression &a, const Expression &b) {
  return {Le(a.impl_, b.impl_)};
}

Expression Lt(const Expression &a, const Expression &b) {
  return {Lt(a.impl_, b.impl_)};
}

Expression Not(const Expression &a) {
  return {Not(a.impl_)};
}

bool ExpectSymbolEq(const Expression &e0, const Expression &e1,
    const std::string &file, const int64_t line) {
  bool res = ExpectSymbolBool(sym::Eq(e0, e1), file, line);
  if (res) {
    GE_ASSERT_SUCCESS(Expression::AppendSymbolEquivalence(e0, e1),
        "[%s:%lld] Append symbol equivalence %s to %s failed",
        file.c_str(), line, e0.Serialize().get(), e1.Serialize().get());
  }
  return res;
}

bool ExpectSymbolBool(const Expression &expr, const std::string &file, const int64_t line) {
  GE_ASSERT_TRUE(expr.IsBooleanExpr(), "Only boolean expr can be use to check symbol, expr: %s",
      expr.Serialize().get());
  if (expr.IsConstExpr()) {
    bool const_value = false;
    GE_ASSERT_TRUE(expr.GetConstValue(const_value));
    return const_value;
  }
  if (GetCurShapeEnvContext() == nullptr) {
    GELOGW("Shape env is nullptr, cannot check symbol, expr: %s", expr.Serialize().get());
    return false;
  }
  bool hint_value = false;
  GE_ASSERT_TRUE(expr.GetHint(hint_value));
  if (hint_value) {
    GE_ASSERT_SUCCESS(GetCurShapeEnvContext()->AppendSymbolCheckInfo(expr, file, line));
  } else {
    GE_ASSERT_SUCCESS(GetCurShapeEnvContext()->AppendSymbolCheckInfo(sym::Not(expr), file, line));
  }
  return hint_value;
}

bool AssertSymbolEq(const Expression &e0, const Expression &e1,
    const std::string &file, const int64_t line) {
  GE_ASSERT_TRUE(AssertSymbolBool(ge::sym::Eq(e0, e1), file, line));
  GE_ASSERT_SUCCESS(Expression::AppendSymbolEquivalence(e0, e1),
      "[%s:%lld] Append symbol equivalence %s to %s failed",
      file.c_str(), line, e0.Serialize().get(), e1.Serialize().get());
  return true;
}

bool AssertSymbolBool(const Expression &expr, const std::string &file, const int64_t line) {
  GE_ASSERT_TRUE(expr.IsBooleanExpr(), "[%s:%lld] Only boolean expr can be used to assert, expr: %s",
      file.c_str(), line, expr.Serialize().get());
  if (expr.IsConstExpr()) {
    bool const_value = false;
    GE_ASSERT_TRUE(expr.GetConstValue(const_value));
    GE_ASSERT_TRUE(const_value, "[%s:%lld] Assert %s failed",
        file.c_str(), line, expr.Serialize().get());
    return const_value;
  }
  if (GetCurShapeEnvContext() == nullptr) {
    GELOGW("Shape env is nullptr, cannot check symbol, expr: %s", expr.Serialize().get());
    return false;
  }
  bool hint_value = false;
  GE_ASSERT_TRUE(expr.GetHint(hint_value));
  GE_ASSERT_TRUE(hint_value, "[%s:%lld] Assert %s failed", file.c_str(), line, expr.Serialize().get());
  GE_ASSERT_SUCCESS(GetCurShapeEnvContext()->AppendSymbolAssertInfo(expr, file, line));
  return true;
}
}  // namespace sym
}  // namespace ge