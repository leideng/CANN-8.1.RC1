/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

/*!
 * \file kernel_bf16.h
 * \brief
 */
#ifndef ASCENDC_BF16_H
#define ASCENDC_BF16_H
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace bfloat16 {
// BF16
constexpr uint32_t BF16_SIGN_INDEX = 15;
constexpr uint32_t BF16_MAN_LEN = 7;
constexpr int16_t BF16_MAX_MAN = 0x07F;
constexpr int16_t BF16_EXP_MASK = 0x7F80;
constexpr uint16_t BF16_ABS_MAX = 0x7FFFu;
constexpr int8_t BF16_MAN_MASK = 0x7F;
constexpr uint16_t BF16_MAN_HIDE_BIT = 0x0080u;
constexpr uint16_t BF16_INFINITY = 0x7C00;
constexpr uint16_t BF16_NAN = 0x7FFF;
// FP32
constexpr uint32_t FP32_SIGN_INDEX = 31;
constexpr uint32_t FP32_MAN_LEN = 23;
constexpr uint32_t FP32_EXP_BIAS = 127;
/*
 * @ingroup fp32 basic parameter
 * @brief   sign mask of fp32         (1 0000 0000  0000 0000 0000 0000 000)
 */
#define FP32_SIGN_MASK (0x80000000u)
/*
 * @ingroup fp32 basic parameter
 * @brief   exponent mask of fp32     (  1111 1111  0000 0000 0000 0000 000)
 */
#define FP32_EXP_MASK (0x7F800000u)
/*
 * @ingroup fp32 basic parameter
 * @brief   mantissa mask of fp32     (             1111 1111 1111 1111 111)
 */
#define FP32_MAN_MASK (0x007FFFFFu)
/*
 * @ingroup fp32 basic parameter
 * @brief   hidd bit of mantissa of fp32      (  1  0000 0000 0000 0000 000)
 */
#define FP32_MAN_HIDE_BIT (0x00800000u)
#define FP32_POS_INFINITY (0x7F800000)

constexpr uint32_t FP32_NAN = 0x7FFFFFFF;
#define BF16_EXP_BIAS FP32_EXP_BIAS
#define BF16_MAX_EXP FP32_MAX_EXP
#define BF16_SIGN_INDEX (15)

union ConvertU32ToFp32 {
    uint32_t i;
    float f;
};

inline uint16_t Bf16Constructor(uint16_t s, uint16_t e, uint16_t m)
{
    return (((s) << BF16_SIGN_INDEX) | ((e) << BF16_MAN_LEN) | ((m) & BF16_MAX_MAN));
}

inline bool Bf16IsNan(const uint16_t& x)
{
    return ((((x) & BF16_EXP_MASK) == BF16_EXP_MASK) && (((x) & BF16_MAN_MASK) != 0));
}

inline bool Bf16IsInf(const uint16_t& x)
{
    return ((((x) & BF16_EXP_MASK) == BF16_EXP_MASK) && (((x) & BF16_MAN_MASK) == 0));
}

inline bool Bf16IsInvalid(uint16_t x)
{
    return ((x & BF16_EXP_MASK) == BF16_EXP_MASK);
}

inline uint16_t Bf16ExtracSign(uint16_t x)
{
    return (((x) >> BF16_SIGN_INDEX) & 1);
}

inline uint16_t Bf16ExtracExp(uint16_t x)
{
    return (((x) & BF16_EXP_MASK) >> BF16_MAN_LEN);
}

inline bool Fp32IsInf(const uint32_t& x)
{
    return ((((x) & FP32_EXP_MASK) == FP32_EXP_MASK) && (((x) & FP32_MAN_MASK) == 0));
}
/*
 * @ingroup fp32 special value judgment
 * @brief   whether a fp32 is NaN
 */
inline bool Fp32IsNan(const uint32_t& x)
{
    return ((((x) & FP32_EXP_MASK) == FP32_EXP_MASK) && (((x) & FP32_MAN_MASK) != 0));
}
/*
 * @ingroup fp32 basic operator
 * @brief   get sign of fp32
 */
inline uint16_t Fp32ExtracSign(uint32_t x)
{
    return (((x) >> FP32_SIGN_INDEX) & 1);
}
/*
 * @ingroup fp32 basic operator
 * @brief   get exponent of Bf16
 */
inline uint32_t Fp32ExtracExp(uint32_t x)
{
    return (((x) & FP32_EXP_MASK) >> FP32_MAN_LEN);
}
/*
 * @ingroup Bf16T
 * @brief   value range is the same as float, but poor accuracy compared to float
 * bit15:       1 bit SIGN      +---+-----+------------+
 * bit14-7:     8 bit EXP       | S |EEEEEEEE|MMM MMMM|
 * bit0-6:      7 bit MAN       +---+-----+------------+
 */
struct Bf16T {
    uint16_t val;
public:
    Bf16T(void) : val(0x0u) {}
    Bf16T(const Bf16T& bf) : val(bf.val) {}
    Bf16T(const float& fVal) : val(FloatToBf16(fVal)) {}
    uint16_t FloatToBf16(const float& fVal) const;
    /*
     * @ingroup Bf16T math evaluation operator
     * @param [in] fp Bf16T object to be copy to Bf16T
     * @brief   Override basic evaluation operator to copy Bf16T to a new Bf16T
     * @return  Return Bf16T result from fp
     */
    Bf16T& operator = (const Bf16T& fp);
    /*
     * @ingroup Bf16T math evaluation operator
     * @param [in] fVal float object to be converted to Bf16T
     * @brief   Override basic evaluation operator to convert float to Bf16T
     * @return  Return Bf16T result from fVal
     */
    Bf16T& operator = (const float& fVal);
    /*
     * @ingroup Bf16T math operator
     * @param [in] fp Bf16T object to be added
     * @brief   Override addition operator to performing Bf16T addition
     * @return  Return Bf16T result of adding this and fp
     */
    Bf16T operator + (const Bf16T fp) const;
    /*
     * @ingroup Bf16T math operator
     * @param [in] fp Bf16T object to be added
     * @brief   Override addition operator to performing Bf16T addition
     * @return  Return Bf16T result of adding this and fp
     */
    Bf16T operator += (const Bf16T fp);
    /*
     * @ingroup Bf16T math conversion
     * @brief   Override convert operator to convert Bf16T to float/fp32
     * @return  Return float/fp32 value of Bf16T
     */
    operator float() const;
    /*
     * @ingroup Bf16T math conversion
     * @brief   Override convert operator to convert Bf16T to double/fp64
     * @return  Return double/fp64 value of Bf16T
     */
    float ToFloat() const;
    uint16_t Bf16Add(uint16_t fp1, uint16_t fp2) const;
};
} // namespace bfloat16
using bfloat16_t = bfloat16::Bf16T;
#endif // ASCENDC_BF16_H

