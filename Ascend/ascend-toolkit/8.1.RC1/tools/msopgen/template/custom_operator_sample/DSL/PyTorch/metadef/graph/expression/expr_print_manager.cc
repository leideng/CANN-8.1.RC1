/* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "expr_print_manager.h"

#include <set>
#include <symengine/basic.h>
#include <symengine/constants.h>
#include <symengine/rational.h>
#include <symengine/symengine_casts.h>
#include <symengine/pow.h>

#include "const_values.h"
#include "common/checker.h"

namespace ge {
namespace {
const std::string kPrintAdd = " + ";
const std::string kPrintSub = " - ";
const std::string kPrintMul = " * ";
const std::string kPrintDiv = " / ";
const std::string kPrintMod = " % ";
const std::string kPrintEq = " == ";
const std::string kPrintNe = " != ";
const std::string kPrintLe = " <= ";
const std::string kPrintLt = " < ";
const std::string kPrintPow = "Pow";
const std::string kPrintLog = "Log";
const std::string kPrintMax = "Max";
const std::string kPrintMin = "Min";
const std::string kPrintExp = "Exp";
const std::string kPrintSqrt = "Sqrt";
const std::string kPrintCeil = "Ceiling";
const std::string kPrintAbs = "Abs";
const std::string kPrintDelim = ", ";
const std::string kPrintBracket_L = "(";
const std::string kPrintBracket_R = ")";
const size_t kRelationArgsNum = 2UL;
}

std::string PrintArgs(const std::vector<SymEngineExprPtr> &args, const std::string &delim) {
  std::string res;
  std::vector<std::string> args_str;
  for (size_t i = 0u; i < args.size(); ++i) {
    args_str.emplace_back(ExpressionImpl::SymExprToExpressionImplRef(args[i]).Str());
  }
  // 保证序列化反序列化后的顺序
  std::sort(args_str.begin(), args_str.end());
  for (size_t i = 0u; i < args_str.size(); ++i) {
    if (i > 0u) {
      res += delim + args_str[i];
      continue;
    }
    res = args_str[i];
  }
  return res;
}

std::string DefaultCeilPrinter(const std::vector<SymEngineExprPtr> &args) {
  return kPrintCeil + kPrintBracket_L + PrintArgs(args, kPrintDelim) + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(kOpCeil, DefaultCeilPrinter);

std::string DefaultAbsPrinter(const std::vector<SymEngineExprPtr> &args) {
  return kPrintAbs + kPrintBracket_L + PrintArgs(args, kPrintDelim) + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(kOpAbs, DefaultAbsPrinter);

std::string DefaultAddPrinter(const std::vector<SymEngineExprPtr> &args) {
  std::vector<SymEngineExprPtr> positive_args;
  std::vector<SymEngineExprPtr> negative_args;
  for (const auto &arg : args) {
    if (SymEngine::is_a<SymEngine::Mul>(*arg) &&
        (SymEngine::down_cast<const SymEngine::Mul&>(*arg)).get_coef()->is_negative()) {
      negative_args.push_back(SymEngine::mul(arg, SymEngine::minus_one));
      continue;
    }
    positive_args.push_back(arg);
  }
  std::string res_str = kPrintBracket_L;
  if (!positive_args.empty()) {
    res_str += PrintArgs(positive_args, kPrintAdd);
  }
  if (!negative_args.empty()) {
    res_str += kPrintSub;
    res_str += PrintArgs(negative_args, kPrintSub);
  }
  res_str += kPrintBracket_R;
  return res_str;
}
REGISTER_EXPR_DEFAULT_PRINTER(kOpAdd, DefaultAddPrinter);

std::string DefaultMulPrinter(const std::vector<SymEngineExprPtr> &args) {
  // split mul to num and dens
  std::vector<SymEngineExprPtr> positive_args;
  std::vector<SymEngineExprPtr> negative_args;
  for (const auto &arg : args) {
    if (SymEngine::is_a<SymEngine::Pow>(*arg)) {
      const auto exp = SymEngine::down_cast<const SymEngine::Pow&>(*arg).get_exp();
      if (SymEngine::is_a_Number(*exp) &&
          SymEngine::down_cast<const SymEngine::Number &>(*exp).is_negative()) {
        negative_args.push_back(SymEngine::div(SymEngine::one, arg));
        continue;
      }
    }
    positive_args.push_back(arg);
  }
  std::string res_str = kPrintBracket_L;
  if (!positive_args.empty()) {
    res_str += PrintArgs(positive_args, kPrintMul);
  } else {
    res_str += std::to_string(sym::kConstOne);
  }
  if (!negative_args.empty()) {
    res_str += kPrintDiv;
    res_str += kPrintBracket_L + PrintArgs(negative_args, kPrintMul) + kPrintBracket_R;
  }
  res_str += kPrintBracket_R;
  return res_str;
}
REGISTER_EXPR_DEFAULT_PRINTER(kOpMul, DefaultMulPrinter);

std::string DefaultMaxPrinter(const std::vector<SymEngineExprPtr> &args) {
  std::string res_str;
  if (args.size() >= kSizeTwo) {
    res_str = kPrintMax + kPrintBracket_L +
                ExpressionImpl::SymExprToExpressionImplRef(args[0]).Str() + kPrintDelim +
                ExpressionImpl::SymExprToExpressionImplRef(args[1]).Str() + kPrintBracket_R;
  }
  for (size_t i = kSizeTwo; i < args.size(); ++i) {
    res_str = kPrintMax + kPrintBracket_L +
                res_str + kPrintDelim + ExpressionImpl::SymExprToExpressionImplRef(args[i]).Str() +
                kPrintBracket_R;
  }
  return res_str;
}
REGISTER_EXPR_DEFAULT_PRINTER(kOpMax, DefaultMaxPrinter);

std::string DefaultMinPrinter(const std::vector<SymEngineExprPtr> &args) {
  std::string res_str;
  if (args.size() >= kSizeTwo) {
    res_str = kPrintMin + kPrintBracket_L
              + ExpressionImpl::SymExprToExpressionImplRef(args[0]).Str() + kPrintDelim +
                ExpressionImpl::SymExprToExpressionImplRef(args[1]).Str() + kPrintBracket_R;
  }
  for (size_t i = kSizeTwo; i < args.size(); ++i) {
    res_str = kPrintMin + kPrintBracket_L +
                res_str + kPrintDelim + ExpressionImpl::SymExprToExpressionImplRef(args[i]).Str() +
                kPrintBracket_R;
  }
  return res_str;
}
REGISTER_EXPR_DEFAULT_PRINTER(kOpMin, DefaultMinPrinter);

std::string PrintIntExpPow(const SymEngineExprPtr &base, const uint32_t exp) {
  std::string res_str = "(";
  for (uint32_t i = 0u; i < exp; ++i) {
    if (i > 0u) {
      res_str += " * " + ExpressionImpl::SymExprToExpressionImplRef(base).Str();
      continue;
    }
    res_str += ExpressionImpl::SymExprToExpressionImplRef(base).Str();
  }
  return res_str + ")";
}

std::string GetDefaultPowPrint(const std::vector<SymEngineExprPtr> &base_args) {
  const size_t base_idx = 0u;
  const size_t exp_idx = 1u;
  return kPrintPow + "(" +
           ExpressionImpl::SymExprToExpressionImplRef(base_args[base_idx]).Str() + ", " +
           ExpressionImpl::SymExprToExpressionImplRef(base_args[exp_idx]).Str() + ")";
}


std::string DefaultPowPrinter(const std::vector<SymEngineExprPtr> &args) {
  const size_t base_idx = 0u;
  const size_t exp_idx = 1u;
  if (args[base_idx]->__eq__(*(SymEngine::E))) {
    return kPrintExp + "(" + ExpressionImpl::SymExprToExpressionImplRef(args[exp_idx]).Str() + ")";
  }
  if (args[exp_idx]->__eq__(*SymEngine::rational(sym::kNumOne, sym::kNumTwo))) {
    return kPrintSqrt + "(" + ExpressionImpl::SymExprToExpressionImplRef(args[base_idx]).Str() + ")";
  }
  if (args[exp_idx]->__eq__(*SymEngine::integer(sym::kNumOne))) {
    return "(" + ExpressionImpl::SymExprToExpressionImplRef(args[base_idx]).Str() + ")";
  }
  if (SymEngine::is_a<SymEngine::Integer>(*(args[exp_idx]))) {
    const SymEngine::Integer &exp_arg =  SymEngine::down_cast<const SymEngine::Integer&>(*(args[exp_idx]));
    if (exp_arg.is_positive()) {
      return PrintIntExpPow(args[base_idx], exp_arg.as_uint());
    }
  }
  return GetDefaultPowPrint(args);
}
REGISTER_EXPR_DEFAULT_PRINTER(kOpPow, DefaultPowPrinter);

std::string DefaultLogPrinter(const std::vector<SymEngineExprPtr> &args) {
  return kPrintLog + kPrintBracket_L + PrintArgs(args, kPrintDelim) + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(kOpLog, DefaultLogPrinter);

std::string DefaultEqualPrinter(const std::vector<SymEngineExprPtr> &args) {
  GE_ASSERT_TRUE(args.size() == kRelationArgsNum,
      "Equal operator args size should be 2, but get %zu", args.size());
  return kPrintBracket_L + ExpressionImpl::SymExprToExpressionImplRef(args[0]).Str() +
      kPrintEq + ExpressionImpl::SymExprToExpressionImplRef(args[1]).Str() + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(kOpEq, DefaultEqualPrinter);

std::string DefaultUnEqualPrinter(const std::vector<SymEngineExprPtr> &args) {
  GE_ASSERT_TRUE(args.size() == kRelationArgsNum,
      "Unequal operator args size should be 2, but get %zu", args.size());
  return kPrintBracket_L + ExpressionImpl::SymExprToExpressionImplRef(args[0]).Str() +
      kPrintNe + ExpressionImpl::SymExprToExpressionImplRef(args[1]).Str() + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(kOpNe, DefaultUnEqualPrinter);

std::string DefaultStrictLessThanPrinter(const std::vector<SymEngineExprPtr> &args) {
  GE_ASSERT_TRUE(args.size() == kRelationArgsNum,
      "StrictLessThan operator args size should be 2, but get %zu", args.size());
  return kPrintBracket_L + ExpressionImpl::SymExprToExpressionImplRef(args[0]).Str() +
      kPrintLt + ExpressionImpl::SymExprToExpressionImplRef(args[1]).Str() + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(kOpLt, DefaultStrictLessThanPrinter);

std::string DefaultLessThanPrinter(const std::vector<SymEngineExprPtr> &args) {
  GE_ASSERT_TRUE(args.size() == kRelationArgsNum,
      "LessThan operator args size should be 2, but get %zu", args.size());
  return kPrintBracket_L + ExpressionImpl::SymExprToExpressionImplRef(args[0]).Str() +
      kPrintLe + ExpressionImpl::SymExprToExpressionImplRef(args[1]).Str() + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(kOpLe, DefaultLessThanPrinter);
}  // namespace ge