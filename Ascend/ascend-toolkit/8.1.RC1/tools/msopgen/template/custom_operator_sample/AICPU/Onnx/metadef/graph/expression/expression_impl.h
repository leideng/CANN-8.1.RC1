/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GRAPH_EXPRESSION_IMPL_H_
#define GRAPH_EXPRESSION_IMPL_H_
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <symengine/basic.h>
#include <symengine/integer.h>
#include "graph/symbolic.h"
#include "graph/debug/ge_util.h"

#define IF_NULL_RETURN_NULL(x)                                                                                         \
  if ((x) == nullptr)                                                                                                  \
  return nullptr

namespace ge {
constexpr int32_t kSizeTwo = 2;

class ExpressionImpl;
using ExpressionImplPtr = std::unique_ptr<ExpressionImpl>;

enum OperationType : size_t {
  kOpAdd = 0,
  kOpMax,
  kOpMin,
  kOpMul,
  kOpAbs,
  kOpPow,
  kOpLog,
  kOpCeil,
  kOpEq,
  kOpNe,
  kOpLt,
  kOpLe,
  kOpNone = std::numeric_limits<size_t>::max()
};

using SymEngineExprPtr = SymEngine::RCP<const SymEngine::Basic>;

class ExpressionImpl {
 public:
  // 目前只支持int32_t,int64_t,uint32_t,uint64_t,const string&,const SymEngineExprPtr&
  template<typename T>
  static std::unique_ptr<ExpressionImpl> CreateExpressionImpl(T value, const std::string &name = "") {
    return ge::ComGraphMakeUnique<ExpressionImpl>(value, name);
  }
  ExpressionImpl() = default;
  ExpressionImpl(int32_t const_value, const std::string &name);
  ExpressionImpl(int64_t const_value, const std::string &name);
  ExpressionImpl(uint32_t const_value, const std::string &name);
  ExpressionImpl(uint64_t const_value, const std::string &name);
  ExpressionImpl(double const_value, const std::string &name);
  ExpressionImpl(bool const_value, const std::string &name);
  explicit ExpressionImpl(const std::string &name);
  ExpressionImpl(const SymEngineExprPtr &sym_expr, const std::string &name);

  static ExpressionImplPtr CreateExpressionImpl(const std::string &name);
  ~ExpressionImpl();

  std::string Str(const StrType type = StrType::kStrCpp) const;
  static ExpressionImplPtr Parse(const std::string &expr_str);
  static ExpressionImplPtr Deserialize(const std::string &expr_str);
  ExprType GetExprType() const;
  bool IsConstExpr() const;
  bool IsVariableExpr() const;
  bool IsBooleanExpr() const;
  ExpressionImplPtr Replace(const std::map<ExpressionImpl *, ExpressionImpl *> &replace_vars) const;
  ExpressionImplPtr Subs(const std::map<ExpressionImpl *, ExpressionImpl *> &subs_vars) const;

  ExpressionImplPtr Simplify() const;
  bool ContainVar(const ExpressionImpl *e) const;
  bool operator==(const ExpressionImpl &e) const;
  std::vector<ExpressionImplPtr> FreeSymbols() const;
  OperationType GetOpType() const;
  std::string GetName() const;
  bool GetResult(double &result) const;

  bool GetConstValue(uint64_t &value) const;
  bool GetConstValue(uint32_t &value) const;
  bool GetConstValue(int32_t &value) const;
  bool GetConstValue(int64_t &value) const;
  bool GetConstValue(bool &value) const;
  bool GetConstValue(double &value) const;
  bool GetConstValue(float &value) const;

  // 该方法不需要new一个ExpressionImpl对象(带来大量的指针校验)就能调用ExpressionImpl的方法
  // ***使用时需注意：1.返回的引用使用时，sym_expr对象必须存在；
  // ***使用时需注意：2.ExpressionImpl类只有一个SymEngineExprPtr类型的私有变量
  static const ExpressionImpl &SymExprToExpressionImplRef(const SymEngineExprPtr &sym_expr) {
    return *reinterpret_cast<const ExpressionImpl *>(&sym_expr);
  }

 private:
  double GetIntegerResult(const SymEngine::Integer &integer_expr) const;

  friend ExpressionImplPtr Add(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Sub(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Mul(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Div(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Max(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Min(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Abs(const ExpressionImplPtr &a);
  friend ExpressionImplPtr Pow(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Log(const ExpressionImplPtr &a);  // 默认以E为底
  friend ExpressionImplPtr Log(const ExpressionImplPtr &arg, const ExpressionImplPtr &base);
  friend ExpressionImplPtr Coeff(const ExpressionImplPtr &b, const ExpressionImplPtr &x, const ExpressionImplPtr &n);
  friend ExpressionImplPtr Ceiling(const ExpressionImplPtr &a);
  friend ExpressionImplPtr Rational(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Eq(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Ne(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Lt(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Le(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Not(const ExpressionImplPtr &a);
  // friend std::string DefaultPowPrinter(const std::vector<ExpressionImplPtr> &args);
  friend class Parser;

 private:
  SymEngineExprPtr sym_expr_;  // 非空,symengine在内存不够时会抛异常
  mutable std::string name_;
};

ExpressionImplPtr Add(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Sub(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Mul(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Div(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Max(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Min(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Abs(const ExpressionImplPtr &a);
ExpressionImplPtr Pow(const ExpressionImplPtr &base, const ExpressionImplPtr &arg);
ExpressionImplPtr Log(const ExpressionImplPtr &arg);  // 默认以E为底
ExpressionImplPtr Log(const ExpressionImplPtr &arg, const ExpressionImplPtr &base);
ExpressionImplPtr Coeff(const ExpressionImplPtr &b, const ExpressionImplPtr &x, const ExpressionImplPtr &n);
ExpressionImplPtr Ceiling(const ExpressionImplPtr &a);
ExpressionImplPtr Rational(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
std::ostream &operator<<(std::ostream &os, const ExpressionImpl &e);
ExpressionImplPtr Eq(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Ne(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Lt(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Le(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Not(const ExpressionImplPtr &a);
}  // namespace ge

#endif  // GRAPH_EXPRESSION_IMPL_H_