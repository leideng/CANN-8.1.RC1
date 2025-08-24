/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "expression_impl.h"

#include <type_traits>
#include <queue>
#include <typeinfo>
#include <symengine/assumptions.h>
#include <symengine/functions.h>
#include <symengine/simplify.h>
#include <symengine/integer.h>
#include <symengine/real_double.h>
#include <symengine/visitor.h>
#include <symengine/logic.h>
#include <symengine/parser.h>

#include "expr_print_manager.h"
#include "const_values.h"
#include "expr_parser.h"
#include "common/checker.h"

namespace ge {
namespace {
constexpr const char_t *kInvalidName = "INVALID_NAME";
}

ExpressionImplPtr ExpressionImpl::CreateExpressionImpl(const std::string &name) {
  return ge::ComGraphMakeUnique<ExpressionImpl>(name);
}

ExpressionImpl::~ExpressionImpl() {}

std::string ExpressionImpl::Str(const StrType type) const {
  if (type == StrType::kStrCpp) {
    if (SymEngine::is_a<SymEngine::Rational>(*sym_expr_)) {
      const auto &x = SymEngine::down_cast<const SymEngine::Rational &>(*sym_expr_);
      auto dens = x.get_den();
      auto nums = x.get_num();
      return "Rational(" + nums->__str__() + " , " + dens->__str__() + ")";
    }
    if (((GetExprType() == ExprType::kExprOperation) ||
        (GetExprType() == ExprType::kExprOperationBoolean)) &&
        (GetOpType() != OperationType::kOpNone)) {
      auto printer = ExprManager::GetInstance().GetPrinter(GetOpType());
      GE_ASSERT_NOTNULL(printer);
      return printer(sym_expr_->get_args());
    }
  }
  return sym_expr_->__str__();
}

ExpressionImplPtr ExpressionImpl::Parse(const std::string &expr_str) {
  Scanner scanner(expr_str);
  ge::ExprParser expr_parser(scanner);
  auto ret = expr_parser.ParserExpression();
  GE_ASSERT_NOTNULL(ret, "Parse expression %s failed.", expr_str.c_str());
  return ret;
}

ExpressionImplPtr ExpressionImpl::Deserialize(const std::string &expr_str) {
  auto ret = Parse(expr_str);
  if (ret->Str() == expr_str) {
    return ret;
  } else {
    GELOGE(ge::PARAM_INVALID, "Parse expression str %s failed, result is %s, please check the string is valid.",
           expr_str.c_str(), ret->Str().c_str());
    return nullptr;
  }
}

ExpressionImplPtr ExpressionImpl::Replace(const std::map<ExpressionImpl *, ExpressionImpl *> &replace_vars) const {
  SymEngine::map_basic_basic sym_replace_vars;
  for (const auto &sym_expr_impl_ptr_item : replace_vars) {
    sym_replace_vars.emplace(sym_expr_impl_ptr_item.first->sym_expr_, sym_expr_impl_ptr_item.second->sym_expr_);
  }
  SymEngineExprPtr replaced_expr = sym_expr_->xreplace(sym_replace_vars);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(replaced_expr);
}

ExpressionImplPtr ExpressionImpl::Subs(const std::map<ExpressionImpl *, ExpressionImpl *> &subs_vars) const {
  SymEngine::map_basic_basic sym_replace_vars;
  for (const auto &sym_expr_impl_ptr_item : subs_vars) {
    sym_replace_vars.emplace(sym_expr_impl_ptr_item.first->sym_expr_, sym_expr_impl_ptr_item.second->sym_expr_);
  }
  SymEngineExprPtr subs_expr = sym_expr_->subs(sym_replace_vars);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(subs_expr);
}

ExpressionImplPtr ExpressionImpl::Simplify() const {
  SymEngineExprPtr simplified_expr = SymEngine::simplify(sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(simplified_expr);
}

bool ExpressionImpl::ContainVar(const ExpressionImpl *e) const {
  if (e->sym_expr_->get_args().size() != 0u) {
    return false;
  }
  for (const auto &arg : FreeSymbols()) {
    if (SymEngine::eq(*arg->sym_expr_, *(e->sym_expr_))) {
      return true;
    }
  }
  return false;
}

std::vector<ExpressionImplPtr> ExpressionImpl::FreeSymbols() const {
  auto free_symbols = SymEngine::free_symbols(*sym_expr_);
  std::vector<ExpressionImplPtr> ret;
  for (const auto &sym_arg : free_symbols) {
    auto expr = ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_arg);
    ret.emplace_back(std::move(expr));
  }
  return ret;
}

bool ExpressionImpl::operator==(const ExpressionImpl &e) const {
  return SymEngine::eq(*sym_expr_, *e.sym_expr_);
}

double ExpressionImpl::GetIntegerResult(const SymEngine::Integer &integer_expr) const {
  if (integer_expr.is_zero()) {
    return 0;
  } else if (integer_expr.is_positive()) {
    return static_cast<double>(integer_expr.as_uint());
  }
  return static_cast<double>(integer_expr.as_int());
}

bool ExpressionImpl::GetResult(double &result) const {
  if (SymEngine::is_a<SymEngine::Integer>(*sym_expr_)) {
    const auto &integer_expr = SymEngine::down_cast<const SymEngine::Integer &>(*sym_expr_);
    result = GetIntegerResult(integer_expr);
    return true;
  }
  if (SymEngine::is_a<SymEngine::Rational>(*sym_expr_)) {
    const auto &rational_expr = SymEngine::down_cast<const SymEngine::Rational &>(*sym_expr_);
    result = GetIntegerResult(*(rational_expr.get_num())) / GetIntegerResult(*(rational_expr.get_den()));
    return true;
  }
  if (SymEngine::is_a<SymEngine::RealDouble>(*sym_expr_)) {
    const auto &real_double_expr = SymEngine::down_cast<const SymEngine::RealDouble &>(*sym_expr_);
    result = real_double_expr.as_double();
    return true;
  }
  return false;
}

bool ExpressionImpl::IsVariableExpr() const {
  return GetExprType() == ExprType::kExprVariable;
}

bool ExpressionImpl::IsBooleanExpr() const {
  return (GetExprType() == ExprType::kExprOperationBoolean) ||
      (GetExprType() == ExprType::kExprConstantBoolean);
}

bool ExpressionImpl::GetConstValue(uint32_t &value) const {
  uint64_t result = 0UL;
  GE_ASSERT_TRUE(GetConstValue(result));
  value = static_cast<uint32_t>(result);
  return true;
}

bool ExpressionImpl::GetConstValue(uint64_t &value) const {
  // 无符号整数类型
  GE_ASSERT_TRUE(SymEngine::is_a<SymEngine::Integer>(*sym_expr_),
      "Cannot get const uint value from a expression: %s not Integer.", Str().c_str());
  const auto &integer_expr = SymEngine::down_cast<const SymEngine::Integer &>(*sym_expr_);
  value = integer_expr.as_uint();
  return true;
}

bool ExpressionImpl::GetConstValue(int32_t &value) const {
  int64_t result = 0L;
  GE_ASSERT_TRUE(GetConstValue(result));
  value = static_cast<int32_t>(result);
  return true;
}

bool ExpressionImpl::GetConstValue(int64_t &value) const {
  // 整数类型
  GE_ASSERT_TRUE(SymEngine::is_a<SymEngine::Integer>(*sym_expr_),
      "Cannot get const int value from a expression: %s not Integer.", Str().c_str());
  const auto &integer_expr = SymEngine::down_cast<const SymEngine::Integer &>(*sym_expr_);
  value = integer_expr.as_int();
  return true;
}

bool ExpressionImpl::GetConstValue(bool &value) const {
  // bool类型
  GE_ASSERT_TRUE(SymEngine::is_a<SymEngine::BooleanAtom>(*sym_expr_),
      "Cannot get const bool value from a expression: %s not BooleanAtom.", Str().c_str());
  const auto &bool_expr = SymEngine::down_cast<const SymEngine::BooleanAtom &>(*sym_expr_);
  value = bool_expr.get_val();
  return true;
}

bool ExpressionImpl::GetConstValue(float &value) const {
  double result = 0L;
  GE_ASSERT_TRUE(GetConstValue(result));
  value = static_cast<float>(result);
  return true;
}

bool ExpressionImpl::GetConstValue(double &value) const {
  GE_ASSERT_TRUE((SymEngine::is_a<SymEngine::RealDouble>(*sym_expr_)) ||
      (SymEngine::is_a<SymEngine::Rational>(*sym_expr_)),
      "Cannot get const float value from a expression: %s not RealDouble or Rational.",
      Str().c_str());
  if (SymEngine::is_a<SymEngine::RealDouble>(*sym_expr_)) {
    const auto &real_double_expr = SymEngine::down_cast<const SymEngine::RealDouble &>(*sym_expr_);
    value = real_double_expr.as_double();
  } else {
    // 分数
    const auto &rational_expr = SymEngine::down_cast<const SymEngine::Rational &>(*sym_expr_);
    value = GetIntegerResult(*(rational_expr.get_num())) / GetIntegerResult(*(rational_expr.get_den()));
  }
  return true;
}

OperationType ExpressionImpl::GetOpType() const {
  if (SymEngine::is_a<SymEngine::Add>(*sym_expr_)) {
    return OperationType::kOpAdd;
  }
  if (SymEngine::is_a<SymEngine::Mul>(*sym_expr_)) {
    return OperationType::kOpMul;
  }
  if (SymEngine::is_a<SymEngine::Max>(*sym_expr_)) {
    return OperationType::kOpMax;
  }
  if (SymEngine::is_a<SymEngine::Min>(*sym_expr_)) {
    return OperationType::kOpMin;
  }
  if (SymEngine::is_a<SymEngine::Pow>(*sym_expr_)) {
    return OperationType::kOpPow;
  }
  if (SymEngine::is_a<SymEngine::Log>(*sym_expr_)) {
    return OperationType::kOpLog;
  }
  if (SymEngine::is_a<SymEngine::Abs>(*sym_expr_)) {
    return OperationType::kOpAbs;
  }
  if (SymEngine::is_a<SymEngine::Ceiling>(*sym_expr_)) {
    return OperationType::kOpCeil;
  }
  if (SymEngine::is_a<SymEngine::Equality>(*sym_expr_)) {
    return OperationType::kOpEq;
  }
  if (SymEngine::is_a<SymEngine::Unequality>(*sym_expr_)) {
    return OperationType::kOpNe;
  }
  if (SymEngine::is_a<SymEngine::LessThan>(*sym_expr_)) {
    return OperationType::kOpLe;
  }
  if (SymEngine::is_a<SymEngine::StrictLessThan>(*sym_expr_)) {
    return OperationType::kOpLt;
  }
  return OperationType::kOpNone;
}

std::string ExpressionImpl::GetName() const {
  if (IsConstExpr() || GetExprType() == ExprType::kExprVariable) {
    if (name_.empty()) {
      static std::atomic<size_t> unique_id(0);
      // 此处不应该使用Str()拼接，比如对于1.0,会生成Const_1.0_1，如果codegen采用此名字定义c++变量会编译报错
      name_ = "Const_" + std::to_string(unique_id.fetch_add(1));
    }
    return name_;
  } else {
    return kInvalidName;
  }
}

ExprType ExpressionImpl::GetExprType() const {
  if (SymEngine::is_a_Number(*sym_expr_)) {
    if (SymEngine::is_a<SymEngine::Integer>(*sym_expr_)) {
      return ExprType::kExprConstantInteger;
    } else if (SymEngine::is_a<SymEngine::RealDouble>(*sym_expr_)) {
      return ExprType::kExprConstantRealDouble;
    } else if (SymEngine::is_a<SymEngine::Rational>(*sym_expr_)) {
      return ExprType::kExprConstantRation;
    } else {
      GELOGE(ge::PARAM_INVALID, "Unsupported type for expression %s", sym_expr_->__str__().c_str());
      return ExprType::kExprNone;
    }
  }
  if (SymEngine::is_a<SymEngine::BooleanAtom>(*sym_expr_)) {
    return ExprType::kExprConstantBoolean;
  }
  if (SymEngine::is_a<SymEngine::Symbol>(*sym_expr_)) {
    return ExprType::kExprVariable;
  }
  if (SymEngine::is_a_Boolean(*sym_expr_)) {
    return ExprType::kExprOperationBoolean;
  }
  return ExprType::kExprOperation;
}

bool ExpressionImpl::IsConstExpr() const {
  return GetExprType() < ExprType::kExprVariable;
}

ExpressionImpl::ExpressionImpl(int32_t const_value, const std::string &name)
    : sym_expr_(SymEngine::integer(const_value)), name_(name) {}

ExpressionImpl::ExpressionImpl(int64_t const_value, const std::string &name)
    : sym_expr_(SymEngine::integer(const_value)), name_(name) {}

ExpressionImpl::ExpressionImpl(uint32_t const_value, const std::string &name)
    : sym_expr_(SymEngine::integer(const_value)), name_(name) {}

ExpressionImpl::ExpressionImpl(uint64_t const_value, const std::string &name)
    : sym_expr_(SymEngine::integer(const_value)), name_(name) {}

ExpressionImpl::ExpressionImpl(double const_value, const std::string &name)
    : sym_expr_(SymEngine::real_double(const_value)), name_(name) {}

ExpressionImpl::ExpressionImpl(bool const_value, const std::string &name)
    : sym_expr_(SymEngine::boolean(const_value)), name_(name) {}

ExpressionImpl::ExpressionImpl(const std::string &name) : sym_expr_(SymEngine::symbol(name)), name_(name) {}

ExpressionImpl::ExpressionImpl(const SymEngineExprPtr &sym_expr, const std::string &name)
    : sym_expr_(sym_expr), name_(name) {}

ExpressionImplPtr Add(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  SymEngineExprPtr sym_expr = SymEngine::add(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Sub(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  SymEngineExprPtr sym_expr = SymEngine::sub(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Mul(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  SymEngineExprPtr sym_expr = SymEngine::mul(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Div(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  SymEngineExprPtr sym_expr = SymEngine::div(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Max(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  SymEngineExprPtr sym_expr = SymEngine::max({a->sym_expr_, b->sym_expr_});
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Min(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  SymEngineExprPtr sym_expr = SymEngine::min({a->sym_expr_, b->sym_expr_});
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Abs(const ExpressionImplPtr &a) {
  GE_ASSERT_NOTNULL(a);
  SymEngineExprPtr sym_expr = SymEngine::abs(a->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Pow(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  SymEngineExprPtr sym_expr = SymEngine::pow(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Log(const ExpressionImplPtr &a) {
  GE_ASSERT_NOTNULL(a);
  SymEngineExprPtr sym_expr = SymEngine::log(a->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Log(const ExpressionImplPtr &arg, const ExpressionImplPtr &base) {
  GE_ASSERT_NOTNULL(arg);
  GE_ASSERT_NOTNULL(base);
  GE_ASSERT_TRUE(!arg->sym_expr_.is_null());
  GE_ASSERT_TRUE(!base->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::log(arg->sym_expr_, base->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Coeff(const ExpressionImplPtr &b, const ExpressionImplPtr &x, const ExpressionImplPtr &n) {
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_NOTNULL(x);
  GE_ASSERT_NOTNULL(n);
  SymEngineExprPtr sym_expr = SymEngine::coeff(*b->sym_expr_.get(), *x->sym_expr_.get(), *n->sym_expr_.get());
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Ceiling(const ExpressionImplPtr &a) {
  GE_ASSERT_NOTNULL(a);
  SymEngineExprPtr sym_expr = SymEngine::ceiling(a->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Rational(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  if (SymEngine::is_a<SymEngine::Integer>(*a->sym_expr_) && SymEngine::is_a<SymEngine::Integer>(*b->sym_expr_)) {
    const auto &integer_expr_a = SymEngine::down_cast<const SymEngine::Integer &>(*a->sym_expr_);
    const auto &integer_expr_b = SymEngine::down_cast<const SymEngine::Integer &>(*b->sym_expr_);
    SymEngineExprPtr sym_expr = SymEngine::Rational::from_two_ints(integer_expr_a, integer_expr_b);
    auto impl = ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
    return impl;
  } else {
    std::cerr << "unsupported rational expr" << std::endl;
    return nullptr;
  }
}

std::ostream &operator<<(std::ostream &os, const ExpressionImpl &e) {
  os << e.Str();
  return os;
}

ExpressionImplPtr Eq(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  SymEngineExprPtr sym_expr = SymEngine::Eq(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Ne(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  SymEngineExprPtr sym_expr = SymEngine::Ne(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Lt(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  SymEngineExprPtr sym_expr = SymEngine::Lt(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}
ExpressionImplPtr Le(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  SymEngineExprPtr sym_expr = SymEngine::Le(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}
ExpressionImplPtr Not(const ExpressionImplPtr &a) {
  GE_ASSERT_NOTNULL(a);
  if (!SymEngine::is_a_Boolean(*a->sym_expr_)) {
      GELOGE(ge::PARAM_INVALID, "Logic operator Not only can handle Boolean expression:%s",
          a->Str().c_str());
    return nullptr;
  }
  SymEngineExprPtr sym_expr =
      SymEngine::logical_not(SymEngine::rcp_static_cast<const SymEngine::Boolean>(a->sym_expr_));
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}
}  // namespace ge
