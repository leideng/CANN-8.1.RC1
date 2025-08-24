/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GRAPH_SYMBOLIC_H_
#define GRAPH_SYMBOLIC_H_

#include <vector>
#include <limits>
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "graph/type_utils.h"

#define EXPECT_SYMBOL_EQ(e0, e1)                                                                                       \
  ge::sym::ExpectSymbolEq(e0, e1, __FILE__, __LINE__)

#define EXPECT_SYMBOL_NE(e0, e1)                                                                                       \
  EXPECT_SYMBOL_CHECK(ge::sym::Ne(e0, e1), __FILE__, __LINE__)

#define EXPECT_SYMBOL_LT(e0, e1)                                                                                       \
  EXPECT_SYMBOL_CHECK(ge::sym::Lt(e0, e1), __FILE__, __LINE__)

#define EXPECT_SYMBOL_LE(e0, e1)                                                                                       \
  EXPECT_SYMBOL_CHECK(ge::sym::Le(e0, e1), __FILE__, __LINE__)

#define EXPECT_SYMBOL_GT(e0, e1)                                                                                       \
  EXPECT_SYMBOL_CHECK(ge::sym::Gt(e0, e1), __FILE__, __LINE__)

#define EXPECT_SYMBOL_GE(e0, e1)                                                                                       \
  EXPECT_SYMBOL_CHECK(ge::sym::Ge(e0, e1), __FILE__, __LINE__)

#define ASSERT_SYMBOL_EQ(e0, e1)                                                                                       \
  do {                                                                                                                 \
    if (!ge::sym::AssertSymbolEq(e0, e1, __FILE__, __LINE__)) {                                                        \
      return ::ErrorResult();                                                                                          \
    }                                                                                                                  \
  } while (false)

#define ASSERT_SYMBOL_NE(e0, e1)                                                                                       \
  ASSERT_SYMBOL_CHECK(ge::sym::Ne(e0, e1), __FILE__, __LINE__)

#define ASSERT_SYMBOL_LT(e0, e1)                                                                                       \
  ASSERT_SYMBOL_CHECK(ge::sym::Lt(e0, e1), __FILE__, __LINE__)

#define ASSERT_SYMBOL_LE(e0, e1)                                                                                       \
  ASSERT_SYMBOL_CHECK(ge::sym::Le(e0, e1), __FILE__, __LINE__)

#define ASSERT_SYMBOL_GT(e0, e1)                                                                                       \
  ASSERT_SYMBOL_CHECK(ge::sym::Gt(e0, e1), __FILE__, __LINE__)

#define ASSERT_SYMBOL_GE(e0, e1)                                                                                       \
  ASSERT_SYMBOL_CHECK(ge::sym::Ge(e0, e1), __FILE__, __LINE__)

#define EXPECT_SYMBOL_CHECK(expr, file, line)                                                                          \
  ge::sym::ExpectSymbolBool(expr, file, line)

#define ASSERT_SYMBOL_CHECK(expr, file, line)                                                                          \
  do {                                                                                                                 \
    if (!ge::sym::AssertSymbolBool(expr, file, line)) {                                                                \
      return ::ErrorResult();                                                                                          \
    }                                                                                                                  \
  } while (false)

namespace ge {
class Expression;
class ExpressionImpl;
class ShapeEnvAttr;
using ExpressionImplPtr = std::unique_ptr<ExpressionImpl>;

namespace sym {
Expression Add(const Expression &a, const Expression &b);
Expression Sub(const Expression &a, const Expression &b);
Expression Mul(const Expression &a, const Expression &b);
Expression Div(const Expression &a, const Expression &b);
Expression Max(const Expression &a, const Expression &b);
Expression Min(const Expression &a, const Expression &b);
Expression Pow(const Expression &base, const Expression &exp);
Expression Abs(const Expression &a);
Expression Log(const Expression &a);  // 默认以E为底
Expression Log(const Expression &arg, const Expression &base);
Expression Coeff(const Expression &b, const Expression &x, const Expression &n);
Expression Rational(int32_t num, int32_t den);  // 分数
Expression Ceiling(const Expression &a);
Expression Align(const Expression &arg, uint32_t alignment);
Expression Eq(const Expression &a, const Expression &b); // ==
Expression Ne(const Expression &a, const Expression &b); // !=
Expression Ge(const Expression &a, const Expression &b); // >=
Expression Gt(const Expression &a, const Expression &b); // >
Expression Le(const Expression &a, const Expression &b); // <=
Expression Lt(const Expression &a, const Expression &b); // <
Expression Not(const Expression &a); // !
bool ExpectSymbolEq(const Expression &e0, const Expression &e1,
    const std::string &file, const int64_t line);
bool AssertSymbolEq(const Expression &e0, const Expression &e1,
    const std::string &file, const int64_t line);
bool ExpectSymbolBool(const Expression &expr,
    const std::string &file, const int64_t line);
bool AssertSymbolBool(const Expression &expr,
    const std::string &file, const int64_t line);
}  // namespace sym
std::ostream &operator<<(std::ostream &os, const Expression &e);

enum class ExprType : uint32_t {
  kExprConstantInteger = 0,
  kExprConstantRealDouble = 1,
  kExprConstantRation = 2,
  kExprConstantBoolean = 3,
  // add const defination here
  kExprVariable = 100,
  // add variable defination here
  kExprOperation = 200,
  kExprOperationBoolean,
  // add operation defination here
  kExprNone = std::numeric_limits<uint32_t>::max()
};

enum class StrType : size_t {
  kStrCpp = 0,
  kStrEnd = 1,
};

class Expression {
 public:
  Expression();
  ~Expression();
  Expression(const Expression &other);
  Expression(Expression &&other) noexcept;
  Expression &operator=(const Expression &other);
  Expression &operator=(Expression &&other) noexcept;
  /**
   * @brief 获取表达式转换成字符串
   */
  std::unique_ptr<char_t[]> Str(const StrType type = StrType::kStrCpp) const;
  /**
   * @brief 将字符串转换成表达式，与Str接口匹配
   */
  static Expression Parse(const char_t *str);
  /**
   * @brief 序列化，将表达式转换成字符串
   */
  std::unique_ptr<char_t[]> Serialize() const;
  /**
  * @brief 序列化，将表达式转换成字符串
  */
  std::string ToString() const;
  /**
   * @brief 反序列化，与Serialize接口匹配，将字符串转换成表达式，同时会校验字符串格式是否为序列化接口序列化出的字符串，如果不是则会报错
   */
  static Expression Deserialize(const char_t *str);
  /**
   * @brief 获取表达式的类型
   */
  ExprType GetExprType() const;
  /**
   * @brief 是否是ConstExpr类型
   */
  bool IsConstExpr() const;

  /**
   * @brief 是否是Symbol类型
   */
  bool IsVariableExpr() const;

  /**
   * @brief 是否是Bool类型
   */
  bool IsBooleanExpr() const;

  /**
   * @brief 对当前表达式中的表达式进行替换，例如 y= x+2. y.replace({x, 2*x}) -> y = 2*x + 2
   *        注意当前symengine对div sub的表达式替换能力有缺失，需要用户自己保证，例如x/y*z->Replace({{x/y, m}})会替换失败
   * @param pair<Expr first, Epxr second> first为被替换的表达式，second为替换的表达式
   */
  Expression Replace(const std::vector<std::pair<Expression, Expression>> &replace_vars) const;

  /**
   * @brief 对当前表达式中的符号进行替换，如对于表达式expr = x + y，expr.subs({x:2, y:z+1}) -> y + z + 1。于repalce比较功能较单一，
   *        只能替换单一符号，无法处理复杂的表达式
   * @param subs_vars 待替换的符号列表，pair中first为被替换的表达式，second为替换的表达式
   * @return 替换后表达式
   */
  Expression Subs(const std::vector<std::pair<Expression, Expression>> &subs_vars) const;
  /**
   * @brief 对当前表达式进行化简。例如2+x+4 -> 6+x
   */
  Expression Simplify() const;
  /**
   * @brief 判断当前表达式字符串中是否含有表达式e的子字符串，例如max((x+2), (4*y)) 含有 x和y
   */
  bool ContainVar(const Expression &e) const;

  /**
   * @brief 判断两个Expr是否相等
   */
  bool operator==(const Expression &e) const;
  /**
   * @brief 判断一个expr与常量是否相等
   */
  template<typename T>
  typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
  operator==(const T &e) const;

  /**
   * @brief 判断两个Expr是否不相等
   */
  bool operator!=(const Expression &e) const;

  /**
   * @brief 判断一个expr与常量是否不相等
   */
  template<typename T>
  typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
  operator!=(const T &e) const;

  /**
   * @brief 获取表达式最基础的元素。例如x - (y * z)，返回{x, y, z}, 注意该接口没有依据字符去重
   */
  std::vector<Expression> FreeSymbols() const;

  /**
   * @brief 获取表达式的值
   */
  graphStatus GetResult(const std::vector<std::pair<Expression, Expression>> &vars_value, double &result) const;

  /**
   * @brief 判断表达式是否合法，成员变量impl_为null则不合法
   */
  bool IsValid() const;

  /**
   * @brief 获取常量的值，只有GetExprType为EXPR_CONSTANT时有效
   * @param value 常量的值
   * @return 成功返回true，失败返回false，失败时value的值无效
   */
  template<typename T>
  typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
  GetConstValue(T &value) const;

  /**
   * @brief 基于之前生成的guard信息判断this与e是否相等，仅基于已有guard做校验，不生产新的guard，主要用于内存优化等编译态优化时判断使用
   * @param e 表达式
   */
  bool StaticCheckEq(const Expression &e) const;

  /**
   * @brief 基于之前生成的guard信息判断this与e是否不相等，仅基于已有guard做校验，不生产新的guard，主要用于内存优化等编译态优化时判断使用
   * @param e 表达式
   */
  bool StaticCheckNe(const Expression &e) const;

  /**
   * @brief 基于之前生成的guard信息判断this是否小于e，仅基于已有guard做校验，不生产新的guard，主要用于内存优化等编译态优化时判断使用
   * @param e 表达式
   */
  bool StaticCheckLt(const Expression &e) const;

  /**
   * @brief 基于之前生成的guard信息判断this是否小于等于e，仅基于已有guard做校验，不生产新的guard，主要用于内存优化等编译态优化时判断使用
   * @param e 表达式
   */
  bool StaticCheckLe(const Expression &e) const;

  /**
   * @brief 基于之前生成的guard信息判断this是否大于e，仅基于已有guard做校验，不生产新的guard，主要用于内存优化等编译态优化时判断使用
   * @param e 表达式
   */
  bool StaticCheckGt(const Expression &e) const;

  /**
   * @brief 基于之前生成的guard信息判断this是否大于等于e，仅基于已有guard做校验，不生产新的guard，主要用于内存优化等编译态优化时判断使用
   * @param e 表达式
   */
  bool StaticCheckGe(const Expression &e) const;

  /**
   * @brief 获取表达式hint值
   * @param hint 获取表达式的hint值
  */
  template<typename T>
  typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
  GetHint(T &hint) const {
    return ComputeHint(hint);
  }

  Expression operator+(const Expression &other) const;
  Expression operator-(const Expression &other) const;
  Expression operator*(const Expression &other) const;
  Expression operator/(const Expression &other) const;

  friend Expression sym::Add(const Expression &a, const Expression &b);
  friend Expression sym::Sub(const Expression &a, const Expression &b);
  friend Expression sym::Mul(const Expression &a, const Expression &b);
  friend Expression sym::Div(const Expression &a, const Expression &b);
  friend Expression sym::Max(const Expression &a, const Expression &b);
  friend Expression sym::Min(const Expression &a, const Expression &b);
  friend Expression sym::Pow(const Expression &a, const Expression &b);
  friend Expression sym::Abs(const Expression &a);
  friend Expression sym::Log(const Expression &a);  // 默认以E为底
  friend Expression sym::Log(const Expression &arg, const Expression &base);
  friend Expression sym::Coeff(const Expression &b, const Expression &x, const Expression &n);
  friend Expression sym::Rational(int32_t num, int32_t den);  // 分数
  friend Expression sym::Ceiling(const Expression &a);
  friend Expression sym::Align(const Expression &arg, uint32_t alignment);
  friend std::ostream &operator<<(std::ostream &os, const Expression &e);
  friend Expression sym::Eq(const Expression &a, const Expression &b); // ==
  friend Expression sym::Ne(const Expression &a, const Expression &b); // !=
  friend Expression sym::Ge(const Expression &a, const Expression &b); // >=
  friend Expression sym::Gt(const Expression &a, const Expression &b); // >
  friend Expression sym::Le(const Expression &a, const Expression &b); // <=
  friend Expression sym::Lt(const Expression &a, const Expression &b); // <
  friend Expression sym::Not(const Expression &a); // !
  friend bool sym::ExpectSymbolEq(const Expression &e0, const Expression &e1,
      const std::string &file, const int64_t line);
  friend bool sym::AssertSymbolEq(const Expression &e0, const Expression &e1,
      const std::string &file, const int64_t line);
  friend bool sym::ExpectSymbolBool(const Expression &expr,
      const std::string &file, const int64_t line);
  friend bool sym::AssertSymbolBool(const Expression &expr,
      const std::string &file, const int64_t line);
  friend class ShapeEnvAttr;
 protected:
  Expression(ExpressionImplPtr &&e);
  static bool StaticCheckBool(const Expression &expr);
  static graphStatus AppendSymbolEquivalence(const Expression &e0, const Expression &e1);
  template<typename T>
  typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
  ComputeHint(T &hint) const;
  ExpressionImplPtr impl_;
};

class Symbol : public Expression {
 public:
  // 拷贝构造、赋值、移动构造、移动赋值默认使用基类，需要保证Symbol类大小与Expression类大小一致
  /**
   * @brief 创建常量
   * @param value 常量的值
   * @param name 常量的名称，默认为空，内部不持有该指针
   */
  explicit Symbol(int32_t value, const char_t *name = "");
  explicit Symbol(int64_t value, const char_t *name = "");
  explicit Symbol(uint32_t value, const char_t *name = "");
  explicit Symbol(uint64_t value, const char_t *name = "");
  explicit Symbol(double value, const char_t *name = "");

  /**
   * @brief 创建变量
   * @param name 变量的名称
   */
  explicit Symbol(const char_t *name = "");

  /**
   * @brief 获取symbol的name，返回值是一个unique_ptr，需要用户自己释放
   */
  std::unique_ptr<char_t[]> GetName() const;
  friend class ShapeEnvAttr;
 private:
  explicit Symbol(ExpressionImplPtr &&e);
};

template<typename T>
typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
Expression::operator==(const T &e) const {
  Symbol symbol(e);
  return (*this == symbol);
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
Expression::operator!=(const T &e) const {
  Symbol symbol(e);
  return !(*this == symbol);
}

// 为了保证ABI兼容性，禁用虚函数，Symbol的大小必须和Expression一样
static_assert(sizeof(Symbol) == sizeof(Expression),
              "The size of the subclass Symbol must be equal to the size of the base class Expression.");
template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY inline TypeId GetTypeId<Expression>() {
  return reinterpret_cast<TypeId>(1024);
}
}  // namespace ge
#endif  // GRAPH_SYMBOLIC_H_