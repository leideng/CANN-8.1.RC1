/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
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

#include "hif8.h"
#include <cmath>

namespace aicpu {
namespace hif8_impl {

const float kHif8OverValue = 1.5 * pow(2.0, 15); // hif8可以表示的最大值
constexpr int8_t kBinaryBaseValue = 2; // 2进制基地
constexpr int8_t kHif8ExpMin = -23; // hif8 格式最小指数值
constexpr int8_t kHif8DenormalExpMin = -22; // hif8 格式DENORMALE最小指数值
constexpr int8_t kHif8DenormalExpMax = -15; // hif8 格式DENORMALE最大指数值
constexpr int8_t kHif8D1Exp = 1; // hif8 格式D1指数值
constexpr int8_t kHif8D2ExpMin = 2; // hif8 格式D2最小指数值
constexpr int8_t kHif8D2ExpMax = 3; // hif8 格式D2最大指数值
constexpr int8_t kHif8D3ExpMin = 4; // hif8 格式D3最小指数值
constexpr int8_t kHif8D3ExpMax = 7; // hif8 格式D3最大指数值
constexpr int8_t kHif8D4ExpMin = 8; // hif8 格式D4最小指数值
constexpr int8_t kHif8D4ExpMax = 15; // hif8 格式D4最大指数值

constexpr int8_t kHif8DenormalDotValue = 0; // hif8 格式DML DOT位4bits值
constexpr int8_t kHif8D0DotValue = 1; // hif8 格式D0 DOT位4bits值
constexpr int8_t kHif8D1DotValue = 2; // hif8 格式D1 DOT位4bits值
constexpr int8_t kHif8D2DotValue = 4; // hif8 格式D2 DOT位4bits值
constexpr int8_t kHif8D3DotValue = 8; // hif8 格式D3 DOT位4bits值
constexpr int8_t kHif8D4DotValue = 12; // hif8 格式D4 DOT位4bits值

constexpr int8_t kHif8ExpBitsOne = 1; // hif8 格式指数位宽1bit
constexpr int8_t kHif8ExpBitsTwo = 2; // hif8 格式指数位宽3bit
constexpr int8_t kHif8ExpBitsThree = 3; // hif8 格式指数位宽3bit
constexpr int8_t kHif8ExpBitsFour = 4; // hif8 格式指数位宽4bit

constexpr int8_t kHif8ManBitsOne = 1; // hif8 格式尾数位宽1bit
constexpr int8_t kHif8ManBitsTwo = 2; // hif8 格式尾数位宽2bit
constexpr int8_t kHif8ManBitsThree = 3; // hif8 格式尾数位宽3bit
constexpr uint8_t kHif8NegSignValue = 128; // hif8 负数时符号位赋值
constexpr int8_t kHif8DotLeftShift = 3; // hif8 DOT位移位位宽

/*
 * 获取hif8_info,为方便后续计算dot存放4bits的value，exponent存放bits位宽，man_bits存放bits位宽
 */
void GetHif8Info(const int8_t exponent, HiF8Info &hif8_info)
{
  KERNEL_LOG_INFO("Input info is exponent[%hd]", exponent);
  if (exponent < kHif8DenormalExpMin) { // -22是hif8表示的最小指数范围值
    // zero
    hif8_info.dot_value = -1;
    hif8_info.exponent_bits = kHif8ExpBitsThree;
    hif8_info.man_bits = 0;
  } else if (exponent >= kHif8DenormalExpMin && exponent < kHif8DenormalExpMax) {
    // dml---dot code: 0000
    hif8_info.dot_value = kHif8DenormalDotValue;
    hif8_info.exponent_bits = kHif8ExpBitsThree;
    hif8_info.man_bits = 0;
  } else if (exponent == 0) {
    // d0---dot code: 0001
    hif8_info.dot_value = kHif8D0DotValue;
    hif8_info.exponent_bits = 0;
    hif8_info.man_bits = kHif8ManBitsThree;
  } else if (abs(exponent) == kHif8D1Exp) {
    // d1---dot code: 001
    hif8_info.dot_value = kHif8D1DotValue; // 4 bits末尾补0，得到0010对应值2
    hif8_info.exponent_bits = kHif8ExpBitsOne;
    hif8_info.man_bits = kHif8ManBitsThree;
  } else if (abs(exponent) >= kHif8D2ExpMin && abs(exponent) <= kHif8D2ExpMax) {
    // d2---dot code: 01
    hif8_info.dot_value = kHif8D2DotValue; // 4 bits末尾补0，得到0100对应值4
    hif8_info.exponent_bits = kHif8ExpBitsTwo;
    hif8_info.man_bits = kHif8ManBitsThree;
  } else if (abs(exponent) >= kHif8D3ExpMin && abs(exponent) <= kHif8D3ExpMax) {
    // d3---dot code: 10
    hif8_info.dot_value = kHif8D3DotValue; // 4 bits末尾补0，得到1000对应值8
    hif8_info.exponent_bits = kHif8ExpBitsThree;
    hif8_info.man_bits = kHif8ManBitsTwo;
  } else if (abs(exponent) >= kHif8D4ExpMin && abs(exponent) <= kHif8D4ExpMax) {
    // d4---dot code: 11
    hif8_info.dot_value = kHif8D4DotValue; // 4 bits末尾补0，得到1100对应值12
    hif8_info.exponent_bits = kHif8ExpBitsFour;
    hif8_info.man_bits = kHif8ManBitsOne;
  } else if (exponent > kHif8D4ExpMax) {
    // over flow
    hif8_info.dot_value = kHif8D4DotValue; // 4 bits末尾补0，得到1100对应值12
    hif8_info.exponent_bits = kHif8ExpBitsFour;
    hif8_info.man_bits = -1;
  } else {
    KERNEL_LOG_WARN("exponent[%hd] out of range.", exponent);
  }
  KERNEL_LOG_INFO("Get Hif8Info is is dot_value[%hd] exponent_bits[%hd] man_bits[%hd]",
                  hif8_info.dot_value, hif8_info.exponent_bits, hif8_info.man_bits);
  return;
}

/*
 * 获取hif8 尾数部分
 * 尾数舍弃采用四舍五入，要丢弃的最高位是0则全部丢弃，最高位是1，剩余的尾数+1，如果尾数全部进位0，则指数进位
 */
bool FpTaRoundToHif8(const int32_t fraction_int, const int8_t man_bits, const int8_t exponent, const uint32_t mantissa_len, int8_t &hif8_fraction)
{
  KERNEL_LOG_INFO("Input info is fraction_int[%d] man_bits[%hd] exponent[%hd] mantissa_len[%u] hif8_fraction[%hd]",
                  fraction_int, man_bits, exponent, mantissa_len, hif8_fraction);
  if (exponent == kHif8ExpMin) {
    hif8_fraction = 0;
    return true;
  }
  bool carry_exp = false;
  // fp32 fraction is 23, keep man_bits + 1 bits
  int8_t hif8_fraction_tmp = fraction_int >> (mantissa_len - (man_bits + 1));
  if (hif8_fraction_tmp == pow(kBinaryBaseValue, man_bits + 1) - 1) {
    // carry exponent
    carry_exp = true;
    hif8_fraction = 0;
  } else if (hif8_fraction_tmp == 0) {
    // zero
    hif8_fraction = 0;
  } else if (hif8_fraction_tmp % kBinaryBaseValue == 1) { // 2表示2进制表示
    hif8_fraction_tmp += 1;
    hif8_fraction = hif8_fraction_tmp >> 1;
  } else {
    hif8_fraction = hif8_fraction_tmp >> 1;
  }
  KERNEL_LOG_INFO("Get carry_exp[%d], hif8_fraction[%hd]", carry_exp, hif8_fraction);
  return carry_exp;
}

/*
 * 获取hif8
 */
int8_t FpToHif8Proc(const bool sign, const int8_t exponent, const int8_t hif8_fraction, HiF8Info &hif8_info)
{
  KERNEL_LOG_INFO("Input info is sign[%d] exponent[%hd] hif8_fraction[%hd], dot_value[%d]",
                  sign, exponent, hif8_fraction, hif8_info.dot_value);
  int8_t ret = 0;
  if (hif8_info.man_bits == -1) {
    return (sign == true ? kHif8NegMax : kHif8PosMax);
  }
  if (exponent <= -kFp32ManMaxLen) {
    return kHif8Zero;
  }
  int8_t sign_exp = exponent < 0 ? 1 : 0;
  uint8_t sign_int_value = (sign == true) ? kHif8NegSignValue : 0;
  if (hif8_info.dot_value == 0) {
    // dml, x = (-1)^sign * 2^(M - 23)
    ret = sign_int_value + exponent + kFp32ManMaxLen;
  } else if (hif8_info.dot_value == 1) {
    // d0
    int8_t dot_int_value = hif8_info.dot_value << kHif8DotLeftShift;
    ret = sign_int_value + dot_int_value + hif8_fraction;
  } else {
    int8_t abs_exponent = abs(exponent);
    // 指数部分默认的最高位1，需要减去，例如指数为15，15-pow(2,4-1)=7 编码为0111，最高位默认为1
    abs_exponent -= pow(2, hif8_info.exponent_bits - 1);
    int8_t exponent_int_value = abs_exponent << hif8_info.man_bits;
    sign_exp = sign_exp << (hif8_info.exponent_bits - 1 + hif8_info.man_bits); // 指数符号位左移
    int8_t dot_int_value = hif8_info.dot_value << kHif8DotLeftShift;
    ret = sign_int_value + dot_int_value + sign_exp + exponent_int_value + hif8_fraction;
  }
  KERNEL_LOG_INFO("Get Hif8 ret[%hd]", ret);
  return ret;
}

/*
 * FP32->HIF8处理函数
 */
Hif8Raw Fp32ToHif8Rtne(float f)
{
  Hif8Raw ret;
  const uint32_t &x = *(uint32_t *)&f;
  bool sign = Fp32ExtractSign(x);
  float f_abs = fabs(f);
  KERNEL_LOG_INFO("Absolute value of f: %f", f_abs);
  // 特殊值：nan +-inf 0 直接转化
  if (Fp32IsNan(x)) {
    ret.val = kHif8Nan;
    return ret;
  } else if (Fp32IsInf(x) || f_abs >= kHif8OverValue) {
    ret.val = (sign == true ? kHif8NegMax : kHif8PosMax);
    return ret;
  } else if (Fp32IsZero(x)) {
    ret.val = kHif8Zero;
    return ret;
  }
  // 其他场景处理
  // 1、计算exponent值
  int8_t exponent = floor(log2(f_abs));
  KERNEL_LOG_INFO("Computed exponent value: %hd", exponent);
  // 2、根据exponent值，计算hif8_info,包括dot-value e-bits m-bits
  HiF8Info hif8_info = { };
  GetHif8Info(exponent, hif8_info);
  // 3、计算mantissa值，根据公式 f = (-1)^sign * 2^(exponent) * (1 + mantissa/2^23)
  int32_t fraction_int = f_abs * pow(2.0, kFp32ManMaxLen) * pow(2.0, -exponent) - pow(2.0, kFp32ManMaxLen);
  KERNEL_LOG_INFO("Computed fraction value: %d", fraction_int);
  // 4、根据尾数部分，看指数部分是否需要进位及获取尾数值
  int8_t hif8_fraction = 0;
  bool carry_exp = FpTaRoundToHif8(fraction_int, hif8_info.man_bits, exponent, kFp32ManMaxLen, hif8_fraction);
  // 5、若exponet有进位需要重新计算hif8_info
  if (carry_exp) {
    exponent += 1;
    GetHif8Info(exponent, hif8_info);
  }
  // 6、计算hif8值
  ret.val = FpToHif8Proc(sign, exponent, hif8_fraction, hif8_info); 
  return ret;
}

/*
 * FP16->HIF8处理函数
 */
Hif8Raw Fp16ToHif8Rtne(Eigen::half f)
{
  Hif8Raw ret;
  const uint16_t &x = *(uint16_t *)&f;
  bool sign = Fp16ExtractSign(x);
  Eigen::half f_abs = (sign == true ? -f : f);
  KERNEL_LOG_INFO("Absolute value of f: %f", f_abs);
  // 特殊值：nan +-inf 0 直接转化
  if (Fp16IsNan(x)) {
    ret.val = kHif8Nan;
    return ret;
  } else if (Fp16IsInf(x) || f_abs >= kHif8OverValue) {
    ret.val = (sign == true ? kHif8NegMax : kHif8PosMax);
    return ret;
  } else if (Fp16IsZero(x)) {
    ret.val = kHif8Zero;
    return ret;
  }
  // 其他场景处理
  // 1、计算exponent值
  int8_t exponent = floor(log2(static_cast<float>(f_abs)));
  KERNEL_LOG_INFO("Computed exponent value: %hd", exponent);
  // 2、根据exponent值，计算hif8_info,包括dot-value e-bits m-bits
  HiF8Info hif8_info = { };
  GetHif8Info(exponent, hif8_info);
  // 3、计算mantissa值，根据公式 f = (-1)^sign * 2^(exponent) * (1 + mantissa/2^10)
  int32_t fraction_int = f_abs * pow(2.0, kFp16ManMaxLen) * pow(2.0, -exponent) - pow(2.0, kFp16ManMaxLen);
  KERNEL_LOG_INFO("Computed fraction value: %d", fraction_int);
  // 4、根据尾数部分，看指数部分是否需要进位及获取尾数值
  int8_t hif8_fraction = 0;
  bool carry_exp = FpTaRoundToHif8(fraction_int, hif8_info.man_bits, exponent, kFp16ManMaxLen, hif8_fraction);
  // 5、若exponet有进位需要重新计算hif8_info
  if (carry_exp) {
    exponent += 1;
    GetHif8Info(exponent, hif8_info);
  }
  // 6、计算hif8值
  ret.val = FpToHif8Proc(sign, exponent, hif8_fraction, hif8_info);
  return ret;
}

/*
 * BF16->HIF8处理函数
 */
Hif8Raw Bf16ToHif8Rtne(Eigen::bfloat16 f)
{
  Hif8Raw ret;
  const uint16_t &x = *(uint16_t *)&f;
  bool sign = Bf16ExtractSign(x);
  Eigen::bfloat16 f_abs = (sign == true ? -f : f);
  KERNEL_LOG_INFO("Absolute value of f: %f", f_abs);
  // 特殊值：nan +-inf 0 直接转化
  if (Bf16IsNan(x)) {
    ret.val = kHif8Nan;
    return ret;
  } else if (Bf16IsInf(x) || f_abs >= kHif8OverValue) {
    ret.val = (sign == true ? kHif8NegMax : kHif8PosMax);
    return ret;
  } else if (Bf16IsZero(x)) {
    ret.val = kHif8Zero;
    return ret;
  }
  // 其他场景处理
  // 1、计算exponent值
  int8_t exponent = floor(log2(f_abs));
  KERNEL_LOG_INFO("Computed exponent value: %hd", exponent);
  // 2、根据exponent值，计算hif8_info,包括dot-value e-bits m-bits
  HiF8Info hif8_info = { };
  GetHif8Info(exponent, hif8_info);
  // 3、计算mantissa值，根据公式 f = (-1)^sign * 2^(exponent) * (1 + mantissa/2^7)
  int32_t fraction_int = f_abs * pow(2.0, kBf16ManMaxLen) * pow(2.0, -exponent) - pow(2.0, kBf16ManMaxLen);
  KERNEL_LOG_INFO("Computed fraction value: %d", fraction_int);
  // 4、根据尾数部分，看指数部分是否需要进位及获取尾数值
  int8_t hif8_fraction = 0;
  bool carry_exp = FpTaRoundToHif8(fraction_int, hif8_info.man_bits, exponent, kBf16ManMaxLen, hif8_fraction);
  // 5、若exponet有进位需要重新计算hif8_info
  if (carry_exp) {
    exponent += 1;
    GetHif8Info(exponent, hif8_info);
  }
  // 6、计算hif8值
  ret.val = FpToHif8Proc(sign, exponent, hif8_fraction, hif8_info);
  return ret;
}
}
}  // namespace aicpu
