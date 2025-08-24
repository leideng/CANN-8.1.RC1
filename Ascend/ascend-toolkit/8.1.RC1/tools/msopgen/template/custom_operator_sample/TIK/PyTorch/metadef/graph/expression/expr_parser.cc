/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/
#include "expr_parser.h"
#include "common/checker.h"
#include "symengine/real_double.h"
namespace ge {
ExpressionImplPtr ExprParser::ParserExpression() {
  auto ret = ParserRelational();
  GE_ASSERT_NOTNULL(ret);
  return ret;
}

graphStatus ExprParser::Init() {
  GE_ASSERT_SUCCESS(scanner_.GetNextToken(currentToken_));
  return ge::GRAPH_SUCCESS;
}
graphStatus ExprParser::Eat(TokenType type) {
  GE_ASSERT(currentToken_.type == type);
  GE_ASSERT_SUCCESS(scanner_.GetNextToken(currentToken_));
  return ge::GRAPH_SUCCESS;
}

ExpressionImplPtr ExprParser::ParserFactor() {
  switch (currentToken_.type) {
    case TokenType::kIdentifier:
      return ParserIdentifier();
    case TokenType::kLparen:
      return ParserLParen();
    case TokenType::kMax:
      return ParserMaxFunction();
    case TokenType::kMin:
      return ParserMinFunction();
    case TokenType::kPow:
      return ParserPowFunction();
    case TokenType::kLog:
      return ParserLogFunction();
    case TokenType::kCeil:
      return ParserCeilFunction();
    case TokenType::kAbs:
      return ParserAbsFunction();
    case TokenType::kRational:
      return ParserRationalFunction();
    case TokenType::kNumber:
      return ParserNumber();
    case TokenType::kTrue:
    case TokenType::kFalse:
      return ParseConstBoolen();
    default:
      GELOGE(ge::PARAM_INVALID, "Unsupported operator %d when Parser factor.", currentToken_.type);
      return nullptr;
  }
}

ExpressionImplPtr ExprParser::ParserRelational() {
  auto node = ParserAddSubtract();
  GE_ASSERT_NOTNULL(node);
  if (currentToken_.type == TokenType::kEq || currentToken_.type == TokenType::kNe ||
      currentToken_.type == TokenType::kLe || currentToken_.type == TokenType::kLt) {
    TokenType op = currentToken_.type;
    GE_ASSERT_SUCCESS(Eat(op));
    auto right = ParserAddSubtract();
    GE_ASSERT_NOTNULL(right);
    switch (op) {
      case TokenType::kEq:
        node = Eq(node, right);
        break;
      case TokenType::kNe:
        node = Ne(node, right);
        break;
      case TokenType::kLt:
        node = Lt(node, right);
        break;
      case TokenType::kLe:
        node = Le(node, right);
        break;
      default:
        GELOGE(ge::PARAM_INVALID, "unsupported operator %d when Parserr add and sub.", currentToken_.type);
        return nullptr;
    }
  }
  return node;
}

ExpressionImplPtr ExprParser::ParserAddSubtract() {
  auto node = ParserMulDivide();
  GE_ASSERT_NOTNULL(node);
  while (currentToken_.type == TokenType::kPlus || currentToken_.type == TokenType::kMinus) {
    TokenType op = currentToken_.type;
    GE_ASSERT_SUCCESS(Eat(op));
    auto right = ParserMulDivide();
    GE_ASSERT_NOTNULL(right);
    switch (op) {
      case TokenType::kPlus:
        node = Add(node, right);
        break;
      case TokenType::kMinus:
        node = Sub(node, right);
        break;
      default:
        GELOGE(ge::PARAM_INVALID, "unsupported operator %d when Parser add and sub.", currentToken_.type);
        return nullptr;
    }
  }
  return node;
}

ExpressionImplPtr ExprParser::ParserMulDivide() {
  auto node = ParserFactor();
  GE_ASSERT_NOTNULL(node);

  while (currentToken_.type == TokenType::kMultiply || currentToken_.type == TokenType::kDivide) {
    TokenType op = currentToken_.type;
    GE_ASSERT_SUCCESS(Eat(op));
    auto right = ParserFactor();
    GE_ASSERT_NOTNULL(right);
    switch (op) {
      case TokenType::kMultiply:
        node = Mul(node, right);
        GE_ASSERT_NOTNULL(node);
        break;
      case TokenType::kDivide:
        node = Div(node, right);
        GE_ASSERT_NOTNULL(node);
        break;
      default:
        GELOGE(ge::PARAM_INVALID, "unsupported operator %d when Parser mul and divide.", currentToken_.type);
        return nullptr;
    }
  }
  return node;
}

ExpressionImplPtr ExprParser::ParserMaxFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kMax));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserAddSubtract();
  GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
  auto arg2 = ParserAddSubtract();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Max(arg1, arg2);
}

ExpressionImplPtr ExprParser::ParserMinFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kMin));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserAddSubtract();
  GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
  auto arg2 = ParserAddSubtract();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Min(arg1, arg2);
}

ExpressionImplPtr ExprParser::ParserPowFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kPow));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserAddSubtract();
  GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
  auto arg2 = ParserAddSubtract();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Pow(arg1, arg2);
}

ExpressionImplPtr ExprParser::ParserLogFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kLog));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserAddSubtract();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Log(arg1);
}

ExpressionImplPtr ExprParser::ParserCeilFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kCeil));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserAddSubtract();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Ceiling(arg1);
}

ExpressionImplPtr ExprParser::ParserAbsFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kAbs));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserAddSubtract();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Abs(arg1);
}

ExpressionImplPtr ExprParser::ParserRationalFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kRational));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserAddSubtract();
  GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
  auto arg2 = ParserAddSubtract();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Rational(arg1, arg2);
}

ExpressionImplPtr ExprParser::ParserNumber() {
  const std::string &numberStr = currentToken_.value;
  if (numberStr.find('.') != std::string::npos) {
    double value = std::stod(numberStr);
    GE_ASSERT_SUCCESS(Eat(TokenType::kNumber));
    return ExpressionImpl::CreateExpressionImpl(value);  // 返回浮点数节点
  } else {
    int64_t value = std::stoll(numberStr);
    GE_ASSERT_SUCCESS(Eat(TokenType::kNumber));
    return ExpressionImpl::CreateExpressionImpl(value);  // 返回整数节点
  }
}

ExpressionImplPtr ExprParser::ParserIdentifier() {
  const std::string name{currentToken_.value};
  GE_ASSERT_SUCCESS(Eat(TokenType::kIdentifier));
  return ExpressionImpl::CreateExpressionImpl(name);
}

ExpressionImplPtr ExprParser::ParseConstBoolen() {
  bool sym_value = currentToken_.value == "True" ? true : false;
  GE_ASSERT_SUCCESS(Eat(currentToken_.type));
  return ExpressionImpl::CreateExpressionImpl(sym_value);
}

ExpressionImplPtr ExprParser::ParserLParen() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto node = ParserExpression();
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return node;
}
}  // namespace ge