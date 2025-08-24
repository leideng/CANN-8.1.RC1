/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
* ===================================================================================================================*/

#ifndef GRAPH_EXPRESSION_SCANNER_H_
#define GRAPH_EXPRESSION_SCANNER_H_
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <array>
#include "graph/types.h"
#include "graph/ge_error_codes.h"

namespace ge {
enum class TokenType {
  kIdentifier,  // variable names like a, b, c, s0, s1,...
  kNumber,      // numeric constants (if needed)
  kPlus,        // '+'
  kMinus,       // '-'
  kMultiply,    // '*'
  kDivide,      // '/'
  kComma,       // ','
  kLparen,      // '('
  kRparen,      // ')'
  kMax,         // 'std::max'
  kMin,         // 'std::min'
  kPow,         // pow
  kLog,         // log
  kCeil,        // ceil
  kAbs,         // abs
  kRational,    // rational
  kEq,          // ==
  kNe,          // !=
  kLt,          // <
  kLe,          // <=
  kTrue,        // true
  kFalse,       // false
  kEnd          // End of input
};
struct Token {
  TokenType type;
  std::string value;  // For identifiers and numbers
};
class Scanner {
public:
  explicit Scanner(const std::string &input);
  graphStatus GetNextToken(Token &token);

private:
  void Advance(size_t steps = 1);
  void SkipWhitespace();
  std::string ReadIdentifier();
  Token ReadNumber();
  std::string ReadOperator();

  const std::string &input_;
  size_t pos_;
  char_t currentChar_;
};
}  // namespace ge

#endif  // GRAPH_EXPRESSION_SCANNER_H_