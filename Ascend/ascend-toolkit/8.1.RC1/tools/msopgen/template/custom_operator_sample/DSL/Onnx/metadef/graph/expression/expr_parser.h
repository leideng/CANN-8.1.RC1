/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GRAPH_EXPRESSION_PARSER_H_
#define GRAPH_EXPRESSION_PARSER_H_
#include "scanner.h"
#include "graph/debug/ge_util.h"
#include "expression_impl.h"

#include <symengine/integer.h>
#include <symengine/rational.h>
#include <memory>
namespace ge {

class ExprParser {
public:
  explicit ExprParser(Scanner scanner) : scanner_(scanner) {
    Init();
  }
  ExpressionImplPtr ParserExpression();

private:
  graphStatus Init();
  graphStatus Eat(TokenType type);
  ExpressionImplPtr ParserFactor();
  ExpressionImplPtr ParserAddSubtract();
  ExpressionImplPtr ParserMulDivide();
  ExpressionImplPtr ParserRelational();
  ExpressionImplPtr ParserMaxFunction();
  ExpressionImplPtr ParserMinFunction();
  ExpressionImplPtr ParserPowFunction();
  ExpressionImplPtr ParserLogFunction();
  ExpressionImplPtr ParserCeilFunction();
  ExpressionImplPtr ParserAbsFunction();
  ExpressionImplPtr ParserRationalFunction();
  ExpressionImplPtr ParserNumber();
  ExpressionImplPtr ParserIdentifier();
  ExpressionImplPtr ParserLParen();
  ExpressionImplPtr ParseConstBoolen();

  Scanner scanner_;
  Token currentToken_;
};

}
#endif  // GRAPH_EXPRESSION_PARSER_H_