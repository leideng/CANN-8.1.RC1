/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/
#include "scanner.h"
#include <unordered_map>
#include <set>
#include "common/checker.h"

namespace ge {
namespace {
std::unordered_map<std::string, TokenType> tokenMap = {
    {"Max", TokenType::kMax}, {"Min", TokenType::kMin}, {"Pow", TokenType::kPow}, {"Log", TokenType::kLog},
    {"Ceiling", TokenType::kCeil}, {"Rational", TokenType::kRational}, {"+", TokenType::kPlus},
    {"-", TokenType::kMinus}, {"*", TokenType::kMultiply}, {"/", TokenType::kDivide},
    {"(", TokenType::kLparen}, {")", TokenType::kRparen}, {",", TokenType::kComma},
    {"Abs", TokenType::kAbs}, {"==", TokenType::kEq}, {"!=", TokenType::kNe}, {"<=", TokenType::kLe},
    {"<", TokenType::kLt}, {"True", TokenType::kTrue}, {"False", TokenType::kFalse}};
}
Scanner::Scanner(const std::string &input) : input_(input), pos_(0) {
  Advance();
}

graphStatus Scanner::GetNextToken(Token &token) {
  SkipWhitespace();

  if (currentChar_ == '\0') {
    token = {TokenType::kEnd, ""};
    return GRAPH_SUCCESS;
  }

  // 识别数字
  if (std::isdigit(currentChar_)) {
    token = ReadNumber();
    return GRAPH_SUCCESS;
  }
  // 识别函数
  if (std::isalpha(currentChar_)) {
    std::string identifier = ReadIdentifier();
    if (tokenMap.find(identifier) != tokenMap.end()) {
      token = {tokenMap[identifier], identifier};
    } else {
      token = {TokenType::kIdentifier, identifier};
    }
    return GRAPH_SUCCESS;
  }
  // 识别操作符
  const auto token_key = ReadOperator();
  if (tokenMap.find(token_key) != tokenMap.end()) {
    token = {tokenMap[token_key], token_key};
    return GRAPH_SUCCESS;
  } else {
    GELOGE(ge::PARAM_INVALID, "Unsupported operator: %s", token_key.c_str());
    token = {TokenType::kEnd, ""};
    return GRAPH_FAILED;
  }
}

void Scanner::Advance(size_t steps) {
  while (steps-- > 0) {
    if (pos_ < input_.length()) {
      currentChar_ = input_[pos_++];
    } else {
      currentChar_ = '\0';
      break;
    }
  }
}

void Scanner::SkipWhitespace() {
  while (std::isspace(currentChar_)) {
    Advance();
  }
}

std::string Scanner::ReadIdentifier() {
  std::string result;
  while (std::isalnum(currentChar_) || currentChar_ == ':' || currentChar_ == '_') {
    result += currentChar_;
    Advance();
  }
  return result;
}

Token Scanner::ReadNumber() {
  std::string result;
  while (std::isdigit(currentChar_) || currentChar_ == '.') {
    result += currentChar_;
    Advance();
  }
  return {TokenType::kNumber, result};
}

std::string Scanner::ReadOperator() {
  std::string one_char_operator = std::string(1, currentChar_);
  Advance();
  std::string two_char_operator = one_char_operator + currentChar_;
  const std::set<std::string> operator_str = {"==", "!=", "<="};
  if (operator_str.find(two_char_operator) != operator_str.end()) {
    Advance();
    return two_char_operator;
  }
  return one_char_operator;
}
}  // namespace ge