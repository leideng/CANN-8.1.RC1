/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _AICPU_HIF8_H_
#define _AICPU_HIF8_H_

#include "utils/kernel_util.h"
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
namespace hif8_impl {
// FP32
constexpr uint32_t kFp32SignIndex = 31U;
constexpr uint32_t kFp32ExpMask = 0x7F800000U;
constexpr uint32_t kFp32ManMask = 0x007FFFFFU;
constexpr uint32_t kFp32AbsMask = 0x7FFFFFFFU;
// FP16
constexpr uint16_t kFp16SignIndex = 15U;
constexpr uint16_t kFp16ExpMask = 0x7C00U;
constexpr uint16_t kFp16ManMask = 0x03FFU;
constexpr uint16_t kFp16AbsMask = 0x7FFFU;
constexpr uint16_t kFp16MaxExp = 0x001FU;
// BF16
constexpr uint16_t kBf16SignIndex = 15U;
constexpr uint16_t kBf16ExpMask = 0x7F80U;
constexpr uint16_t kBf16ManMask = 0x007FU;
constexpr uint16_t kBf16AbsMask = 0x7FFFU;
constexpr uint16_t kBf16MaxExp = 0x00FFU;
// HIF8
constexpr int8_t kHif8NegMax = 0xEFU;
constexpr int8_t kHif8PosMax = 0x6FU;
constexpr int8_t kHif8Nan = 0x80U;
constexpr int8_t kHif8Zero = 0x00U;

constexpr int8_t kFp32ManMaxLen = 23; // FP32浮点数的最大尾数长度
constexpr int8_t kFp16ManMaxLen = 10; // FP16浮点数的最大尾数长度
constexpr int8_t kBf16ManMaxLen = 7; // BF16浮点数的最大尾数长度
constexpr int8_t kFp32ExpBias = 127; // FP32浮点数的指数偏移
constexpr int8_t kFp16ExpBias = 15; // FP16浮点数的指数偏移
constexpr int8_t kBf16ExpBias = 127; // BF16浮点数的指数偏移

inline bool Fp32IsNan(const uint32_t x)
{
  return ((((x) & kFp32ExpMask) == kFp32ExpMask) && (((x) & kFp32ManMask) != 0));
}

inline bool Fp32IsInf(const uint32_t x)
{
  return ((((x) & kFp32ExpMask) == kFp32ExpMask) && (((x) & kFp32ManMask) == 0));
}

inline bool Fp32IsZero(const uint32_t x)
{
  return (((x) & kFp32AbsMask) == 0);
}

inline bool Fp32ExtractSign(const uint32_t x)
{
  return (((x) >> kFp32SignIndex) & 0x1) == 0x1;
}

inline int8_t Fp32ExtractExp(const uint32_t x)
{
  return (((x) & kFp32ExpMask) >> kFp32ManMaxLen);
}

inline bool Fp16IsNan(const uint16_t x)
{
  return ((((x) & kFp16ExpMask) == kFp16ExpMask) && (((x) & kFp16ManMask) != 0));
}

inline bool Fp16IsInf(const uint16_t x)
{
  return ((((x) & kFp16ExpMask) == kFp16ExpMask) && (((x) & kFp16ManMask) == 0));
}

inline bool Fp16IsZero(const uint16_t x)
{
  return (((x) & kFp16AbsMask) == 0);
}

inline bool Fp16ExtractSign(const uint16_t x)
{
  return (((x) >> kFp16SignIndex) & 0x1) == 0x1;
}

inline int8_t Fp16ExtractExp(const uint16_t x)
{
  return (((x) & kFp16ExpMask) >> kFp16ManMaxLen);
}

inline bool Bf16IsNan(const uint16_t x)
{
  return ((((x) & kBf16ExpMask) == kBf16ExpMask) && (((x) & kBf16ManMask) != 0));
}

inline bool Bf16IsInf(const uint16_t x)
{
  return ((((x) & kBf16ExpMask) == kBf16ExpMask) && (((x) & kBf16ManMask) == 0));
}

inline bool Bf16IsZero(const uint16_t x)
{
  return (((x) & kBf16AbsMask) == 0);
}

inline bool Bf16ExtractSign(const uint16_t x)
{
  return (((x) >> kBf16SignIndex) & 0x1) == 0x1;
}

inline int8_t Bf16ExtractExp(const uint16_t x)
{
  return (((x) & kBf16ExpMask) >> kBf16ManMaxLen);
}

using HiF8Info = struct {
  int8_t dot_value;
  int8_t exponent_bits;
  int8_t man_bits;
};

struct Hif8Raw {
  int8_t val;
  Hif8Raw() : val(0) {}
};

Hif8Raw Fp32ToHif8Rtne(float f);
Hif8Raw Fp16ToHif8Rtne(Eigen::half f);
Hif8Raw Bf16ToHif8Rtne(Eigen::bfloat16 f);

struct Hif8Base : public Hif8Raw {
  Hif8Base() : Hif8Raw() {}
  explicit Hif8Base(const Hif8Raw &v) : Hif8Raw(v) {}
};
} // namespace hif8_impl

struct hif8 : public hif8_impl::Hif8Base {
  explicit hif8() {}
  explicit hif8(const Hif8Raw &v) : hif8_impl::Hif8Base(v) {}

  explicit hif8(float f) : hif8_impl::Hif8Base(hif8_impl::Fp32ToHif8Rtne(f)) {}
  explicit hif8(Eigen::half f) : hif8_impl::Hif8Base(hif8_impl::Fp16ToHif8Rtne(f)) {}
  explicit hif8(Eigen::bfloat16 f) : hif8_impl::Hif8Base(hif8_impl::Bf16ToHif8Rtne(f)) {}
};
}  // namespace aicpu
#endif  // _AICPU_HIF8_H_
