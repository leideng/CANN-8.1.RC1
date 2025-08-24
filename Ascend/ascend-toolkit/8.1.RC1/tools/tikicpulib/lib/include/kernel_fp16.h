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
 * \file kernel_fp16.h
 * \brief
 */
#ifndef ASCENDC_FP16_H
#define ASCENDC_FP16_H
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace float16 {
enum class DimIndex {
    K_DIM0 = 0,
    K_DIM1,
    K_DIM2,
    K_DIM3,
    K_DIM4,
    K_DIM5,
    K_DIM6,
    K_DIM7,
    K_DIM8,
    K_DIM9,
    K_DIM10,
    K_DIM11,
    K_DIM12,
    K_DIM13,
    K_DIM14,
    K_DIM15,
    K_DIM16,
};

enum class BitShift {
    K_BIT_SHIFT2 = 2,
    K_BIT_SHIFT3 = 3,
    K_BIT_SHIFT4 = 4,
    K_BIT_SHIFT5 = 5,
    K_BIT_SHIFT6 = 6,
    K_BIT_SHIFT7 = 7,
    K_BIT_SHIFT8 = 8,
    K_BIT_SHIFT9 = 9,
    K_BIT_SHIFT10 = 10,
    K_BIT_SHIFT11 = 11,
    K_BIT_SHIFT12 = 12,
    K_BIT_SHIFT13 = 13,
    K_BIT_SHIFT14 = 14,
    K_BIT_SHIFT15 = 15,
    K_BIT_SHIFT16 = 16,
    K_BIT_SHIFT20 = 20,
    K_BIT_SHIFT24 = 24,
    K_BIT_SHIFT27 = 27,
    K_BIT_SHIFT28 = 28,
    K_BIT_SHIFT31 = 31,
    K_BIT_SHIFT32 = 32,
    K_BIT_SHIFT36 = 36,
    K_BIT_SHIFT40 = 40,
    K_BIT_SHIFT44 = 44,
    K_BIT_SHIFT48 = 48,
    K_BIT_SHIFT52 = 52,
    K_BIT_SHIFT56 = 56,
    K_BIT_SHIFT59 = 59,
    K_BIT_SHIFT60 = 60,
    K_BIT_SHIFT63 = 63,
    K_BIT_SHIFT64 = 64,
    K_BIT_SHIFT128 = 128,
    K_BIT_SHIFT255 = 255,
    K_BIT_SHIFT256 = 256,
    K_BIT_SHIFT512 = 512,
    K_BIT_SHIFT768 = 768,
    K_BIT_SHIFT784 = 784,
    K_BIT_SHIFT1020 = 1020,
    K_BIT_SHIFT1024 = 1024,
    K_BIT_SHIFT3136 = 3136,
    K_BIT_SHIFT4096 = 4096,
    K_BIT_SHIFT6144 = 6144,
    K_BIT_SHIFT10240 = 10240,
    K_BIT_SHIFT65536 = 65536
};

enum class Fp16BasicParam {
    K_FP16_EXP_BIAS = 15,          // fp16 exponent bias
    K_FP16_EXP_LEN = 5,            // the exponent bit length of fp16 is 5
    K_FP16_MAN_LEN = 10,           // the mantissa bit length of fp16 is 10
    K_FP16_SIGN_INDEX = 15,        // bit index of sign in fp16
    K_FP16_SIGN_MASK = 0x8000,     // sign mask of fp16         (1 00000 00000 00000)
    K_FP16_EXP_MASK = 0x7C00,      // exponent mask of fp16 (  11111 00000 00000)
    K_FP16_MAN_MASK = 0x03FF,      // mantissa mask of fp16 (        11111 11111)
    K_FP16_MAN_HIDE_BIT = 0x0400,  // hide_bit of mantissa of fp16(   1 00000 00000)
    K_FP16_MAX = 0x7BFF,           // maximum value (0111 1011 1111 1111)
    K_FP16_MIN = 0xFBFF,           // minimum value            (1111 1011 1111 1111)
    K_FP16_ABS_MAX = 0x7FFF,       // absolute maximum value   (0111 1111 1111 1111)
    K_FP16_MAX_EXP = 0x001F,       // maximum exponent value of fp16 is 15(11111)
    K_FP16_MAX_VALID_EXP = 0x001E, // maximum valid exponent value of fp16 is 14(11110)
    K_FP16_MAX_MAN = 0x03FF,       // maximum mantissa value of fp16(11111 11111)
};

// / @ingroup fp16 basic operator
// / @brief   get sign of fp16
inline uint16_t FP16_EXTRAC_SIGN(uint16_t x)
{
    return (((x) >> static_cast<uint16_t>(Fp16BasicParam::K_FP16_SIGN_INDEX)) & 1);
}
// / @ingroup fp16 basic operator
// / @brief   get exponent of fp16
inline int16_t FP16_EXTRAC_EXP(uint16_t x)
{
    return (((x) >> static_cast<uint16_t>(Fp16BasicParam::K_FP16_MAN_LEN)) &
        static_cast<uint16_t>(Fp16BasicParam::K_FP16_MAX_EXP));
}
// / @ingroup fp16 basic operator
// / @brief   get mantissa of fp16
inline uint16_t FP16_EXTRAC_MAN(uint16_t x)
{
    return ((((x) >> 0) & 0x3FF) |
        (((((x) >> static_cast<uint16_t>(Fp16BasicParam::K_FP16_MAN_LEN)) & 0x1F) > 0 ? 1 : 0) * 0x400));
}
// / @ingroup fp16 basic operator
// / @brief   constructor of fp16 from sign exponent and mantissa
inline uint16_t FP16_CONSTRUCTOR(uint16_t s, uint16_t e, uint16_t m)
{
    return (((s) << (static_cast<uint16_t>(Fp16BasicParam::K_FP16_SIGN_INDEX))) |
        ((e) << static_cast<uint16_t>(Fp16BasicParam::K_FP16_MAN_LEN)) |
        ((m) & (static_cast<uint16_t>(Fp16BasicParam::K_FP16_MAX_MAN))));
}
// / @ingroup fp16 special value judgment
// / @brief   whether a fp16 is zero
inline bool FP16_IS_ZERO(uint16_t x)
{
    return (x & (static_cast<uint16_t>(Fp16BasicParam::K_FP16_ABS_MAX))) == 0;
}
// / @ingroup fp16 special value judgment
// / @brief   whether a fp16 is a denormalized value
inline bool FP16_IS_DENORM(uint16_t x)
{
    return (x & (static_cast<uint16_t>(Fp16BasicParam::K_FP16_EXP_MASK))) == 0;
}
// / @ingroup fp16 special value judgment
// / @brief   whether a fp16 is invalid
inline bool FP16_IS_INVALID(uint16_t x)
{
    return ((x & static_cast<uint16_t>(Fp16BasicParam::K_FP16_EXP_MASK)) ==
        static_cast<uint16_t>(Fp16BasicParam::K_FP16_EXP_MASK));
}

enum class Fp32BasicParam : uint32_t {
    K_FP32_EXP_BIAS = 127,             // fp32 exponent bias
    K_FP32_EXP_LEN = 8,                // the exponent bit length of float/fp32 is 8
    K_FP32_MAN_LEN = 23,               // the mantissa bit length of float/fp32 is 23
    K_FP32_SIGN_INDEX = 31,            // bit index of sign in float/fp32
    K_FP32_SIGN_MASK = 0x80000000u,    // sign mask of fp32 (1 0000 0000  0000 0000 0000 0000 000)
    K_FP32_EXP_MASK = 0x7F800000u,     // exponent mask of fp32 (  1111 1111  0000 0000 0000 0000 000)
    K_FP32_MAN_MASK = 0x007FFFFFu,     // mantissa mask of fp32 (1111 1111 1111 1111 111)
    K_FP32_MAN_HIDE_BIT = 0x00800000u, // hide_bit of mantissa of fp32 (  1  0000 0000 0000 0000 000)
    K_FP32_ABS_MAX = 0x7FFFFFFFu,      // absolute maximum value (0 1111 1111  1111 1111 1111 1111 111)
    K_FP32_MAX_EXP = 0xFF,             // maximum exponent value of fp32 is 255(1111 1111)
    K_FP32_MAX_MAN = 0x7FFFFF          // maximum mantissa value of fp32 (1111 1111 1111 1111 1111 111)
};

// / @ingroup fp32 basic operator
// / @brief   constructor of fp32 from sign exponent and mantissa
inline uint32_t FP32_CONSTRUCTOR(uint32_t s, uint32_t e, uint32_t m)
{
    return (((s) << static_cast<uint16_t>(Fp32BasicParam::K_FP32_SIGN_INDEX)) |
        ((e) << static_cast<uint16_t>(Fp32BasicParam::K_FP32_MAN_LEN)) |
        ((m) & static_cast<uint32_t>(Fp32BasicParam::K_FP32_MAX_MAN)));
}

enum class Fp64BasicParam : uint64_t {
    K_FP64_EXP_BIAS = 1023,                      // fp64 exponent bias
    K_FP64_EXP_LEN = 11,                         // the exponent bit length of double/fp64 is 11
    K_FP64_MAN_LEN = 52,                         // the mantissa bit length of double/fp64 is 52
    K_FP64_SIGN_INDEX = 63,                      // bit index of sign in double/fp64 is 63
    K_FP64_SIGN_MASK = 0x8000000000000000LLu,    // sign mask of fp64 (1 000 (total 63bits 0))
    K_FP64_EXP_MASK = 0x7FF0000000000000LLu,     // exponent mask of fp64 (0 1 11111 11111  0000?-?-(total 52bits 0))
    K_FP64_MAN_MASK = 0x000FFFFFFFFFFFFFLLu,     // mantissa mask of fp64 ( 1111?-?-(total 52bits 1))
    K_FP64_MAN_HIDE_BIT = 0x0010000000000000LLu, // hide_bit of mantissa of fp64 ( 1 0000?-?-(total 52bits 0))
    K_FP64_ABS_MAX = 0x7FFFFFFFFFFFFFFFLLu,      // absolute maximum value (0 111?-?-(total 63bits 1))
    K_FP64_MAX_EXP = 0x07FF,                     // maximum exponent value of fp64 is 2047(1 11111 11111)
    K_FP64_MAX_MAN = 0xFFFFFFFFFFFLLu            // maximum mantissa value of fp64  (111?-?-(total 52bits 1))
};

enum class NumBitMax : uint64_t {
    K_INT8_MAX = 0x7F,         // maximum positive value of int8_t (0111 1111)
    K_BIT_LEN8_MAX = 0xFF,     // maximum value of a data with 8 bits length  (1111 111)
    K_INT16_MAX = 0x7FFF,      // maximum positive value of int16_t (0111 1111 1111 1111)
    K_BIT_LEN16_MAX = 0xFFFF,  // maximum value of a data with 16 bits length (1111 1111 1111 1111)
    K_INT32_MAX = 0x7FFFFFFFu, // maximum positive value of int32_t (0111 1111 1111 1111 1111 1111 1111 1111)
    // maximum value of uint32_t(1111 1111 1111 1111 1111 1111 1111 1111)
    K_BIT_LEN32_MAX = 0xFFFFFFFFu,
    // maximum value of int64_t  (0111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111)
    K_INT64_MAX = 0x7FFFFFFFFFFFFFFFu,
    // maximum value of uint64_t (1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111)
    K_BIT_LEN64_MAX = 0xFFFFFFFFFFFFFFFFu
};

// / @ingroup Fp16T enum
// / @brief   round mode of last valid digital
enum class TagFp16RoundMode {
    K_ROUND_TO_NEAREST = 0, // < round to nearest even
    K_ROUND_BY_TRUNCATED,   // < round by truncated
    K_ROUND_MODE_RESERVED,
};

#ifndef __TIK_CC
#define __ai_core__
#define __ai_host__
#endif

/**
 * @ingroup Fp16T
 * @brief   Half precision float
 * bit15:       1 bit SIGN      +---+-----+------------+
 * bit14-10:    5 bit EXP       | S |EEEEE|MM MMMM MMMM|
 * bit0-9:      10bit MAN       +---+-----+------------+
 */
#define TIK_ALIGN(n) alignas(n)
// struct TIK_ALIGN(2) Fp16T {
struct Fp16T {
    uint16_t val;

public:
    /* *
     * @ingroup Fp16T constructor
     * @brief   Constructor without any param(default constructor)
     */
    __ai_host__ __ai_core__ Fp16T(void) : val(0x0u) {}
    /* *
     * @ingroup Fp16T copy constructor
     * @brief   Constructor with a Fp16T object(copy constructor)
     */
    __ai_host__ __ai_core__ Fp16T(const Fp16T& fp) : val(fp.val) {}
    /* *
     * @ingroup Fp16T constructor
     * @brief   Constructor with an float value
     */
    __ai_host__ __ai_core__ Fp16T(const float& fVal) : val(FloatToFp16(fVal)) {}
    /* *
     * @ingroup Fp16T constructor
     * @brief   Constructor with an double value
     */
    __ai_host__ __ai_core__ Fp16T(const double& dVal) : val(DoubleToFp16(dVal)) {}
    /* *
     * @ingroup Fp16T constructor
     * @brief   Constructor with an int8_t value
     */
    __ai_host__ __ai_core__ Fp16T(const int8_t& iVal) : val(Int8ToFp16(iVal)) {}
    /* *
     * @ingroup Fp16T constructor
     * @brief   Constructor with an uint8_t value
     */
    __ai_host__ __ai_core__ Fp16T(const uint8_t& uiVal) : val(UInt8ToFp16(uiVal)) {}
    /* *
     * @ingroup Fp16T constructor
     * @brief   Constructor with an int16_t value
     */
    __ai_host__ __ai_core__ Fp16T(const int16_t& iVal) : val(Int16ToFp16(iVal)) {}
    /* *
     * @ingroup Fp16T constructor
     * @brief   Constructor with an uint16_t value
     */
    __ai_host__ __ai_core__ Fp16T(const uint16_t& uiVal) : val(UInt16ToFp16(uiVal)) {}
    /* *
     * @ingroup Fp16T constructor
     * @brief   Constructor with an int32_t value
     */
    __ai_host__ __ai_core__ Fp16T(const int32_t& iVal) : val(Int32ToFp16(iVal)) {}
    /* *
     * @ingroup Fp16T constructor
     * @brief   Constructor with an uint32_t value
     */
    __ai_host__ __ai_core__ Fp16T(const uint32_t& uiVal) : val(UInt32ToFp16(uiVal)) {}

    uint16_t FloatToFp16(const float& fVal) const;
    uint16_t DoubleToFp16(const double& dVal);
    uint16_t Int8ToFp16(const int8_t& iVal) const;
    uint16_t UInt8ToFp16(const uint8_t& uiVal) const;
    uint16_t Int16ToFp16(const int16_t& iVal) const;
    uint16_t UInt16ToFp16(const uint16_t& uiVal);
    uint16_t Int32ToFp16(const int32_t& iVal) const;
    uint16_t UInt32ToFp16(const uint32_t& uiVal) const;

    /* *
     * @ingroup Fp16T math operator
     * @param [in] fp Fp16T object to be added
     * @brief   Override addition operator to performing Fp16T addition
     * @return  Return Fp16T result of adding this and fp
     */
    Fp16T operator + (const Fp16T fp) const;
    /* *
     * @ingroup Fp16T math operator
     * @param [in] fp Fp16T object to be subtracted
     * @brief   Override addition operator to performing Fp16T subtraction
     * @return  Return Fp16T result of subtraction fp from this
     */
    Fp16T operator - (const Fp16T fp) const;
    /* *
     * @ingroup Fp16T math operator
     * @param [in] fp Fp16T object to be multiplied
     * @brief   Override multiplication operator to performing Fp16T
     * multiplication
     * @return  Return Fp16T result of multiplying this and fp
     */
    Fp16T operator*(const Fp16T fp) const;
    /* *
     * @ingroup Fp16T math operator divided
     * @param [in] fp Fp16T object to be divided
     * @brief   Override division operator to performing Fp16T division
     * @return  Return Fp16T result of division this by fp
     */
    Fp16T operator / (const Fp16T fp) const;
    /* *
     * @ingroup Fp16T math operator
     * @param [in] fp Fp16T object to be added
     * @brief   Override addition operator to performing Fp16T addition
     * @return  Return Fp16T result of adding this and fp
     */
    Fp16T operator += (const Fp16T fp);
    /* *
     * @ingroup Fp16T math operator
     * @param [in] fp Fp16T object to be subtracted
     * @brief   Override addition operator to performing Fp16T subtraction
     * @return  Return Fp16T result of subtraction fp from this
     */
    Fp16T operator -= (const Fp16T fp);
    /* *
     * @ingroup Fp16T math operator
     * @param [in] fp Fp16T object to be multiplied
     * @brief   Override multiplication operator to performing Fp16T
     * multiplication
     * @return  Return Fp16T result of multiplying this and fp
     */
    Fp16T operator *= (const Fp16T fp);
    /* *
     * @ingroup Fp16T math operator divided
     * @param [in] fp Fp16T object to be divided
     * @brief   Override division operator to performing Fp16T division
     * @return  Return Fp16T result of division this by fp
     */
    Fp16T operator /= (const Fp16T fp);

    /* *
     * @ingroup Fp16T math compare operator
     * @param [in] fp Fp16T object to be compared
     * @brief   Override basic comparison operator to performing Fp16T if-equal
     * comparison
     * @return  Return boolean result of if-equal comparison of this and fp.
     */
    bool operator == (const Fp16T& fp) const;
    /* *
     * @ingroup Fp16T math compare operator
     * @param [in] fp Fp16T object to be compared
     * @brief   Override basic comparison operator to performing Fp16T not-equal
     * comparison
     * @return  Return boolean result of not-equal comparison of this and fp.
     */
    bool operator != (const Fp16T& fp) const;
    /* *
     * @ingroup Fp16T math compare operator
     * @param [in] fp Fp16T object to be compared
     * @brief   Override basic comparison operator to performing Fp16T
     * greater-than comparison
     * @return  Return boolean result of greater-than comparison of this and fp.
     */
    bool operator > (const Fp16T& fp) const;
    /* *
     * @ingroup Fp16T math compare operator
     * @param [in] fp Fp16T object to be compared
     * @brief   Override basic comparison operator to performing Fp16T
     * greater-equal comparison
     * @return  Return boolean result of greater-equal comparison of this and fp.
     */
    bool operator >= (const Fp16T& fp) const;
    /* *
     * @ingroup Fp16T math compare operator
     * @param [in] fp Fp16T object to be compared
     * @brief   Override basic comparison operator to performing Fp16T less-than
     * comparison
     * @return  Return boolean result of less-than comparison of this and fp.
     */
    bool operator < (const Fp16T& fp) const;
    /* *
     * @ingroup Fp16T math compare operator
     * @param [in] fp Fp16T object to be compared
     * @brief   Override basic comparison operator to performing Fp16T less-equal
     * comparison
     * @return  Return boolean result of less-equal comparison of this and fp.
     */
    bool operator <= (const Fp16T& fp) const;

    /* *
     * @ingroup Fp16T math evaluation operator
     * @param [in] fp Fp16T object to be copy to Fp16T
     * @brief   Override basic evaluation operator to copy Fp16T to a new Fp16T
     * @return  Return Fp16T result from fp
     */
    Fp16T& operator = (const Fp16T& fp);

    /* *
     * @ingroup Fp16T math evaluation operator
     * @param [in] fVal float object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert float to Fp16T
     * @return  Return Fp16T result from fVal
     */
    Fp16T& operator = (const float& fVal);
    /* *
     * @ingroup Fp16T math evaluation operator
     * @param [in] dVal double object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert double to Fp16T
     * @return  Return Fp16T result from dVal
     */
    Fp16T& operator = (const double& dVal);
    /* *
     * @ingroup Fp16T math evaluation operator
     * @param [in] iVal float object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert float to Fp16T
     * @return  Return Fp16T result from iVal
     */
    Fp16T& operator = (const int8_t& iVal);
    /* *
     * @ingroup Fp16T math evaluation operator
     * @param [in] uiVal uint8_t object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert uint8_t to Fp16T
     * @return  Return Fp16T result from uiVal
     */
    Fp16T& operator = (const uint8_t& uiVal);
    /* *
     * @ingroup Fp16T math evaluation operator
     * @param [in] iVal int16_t object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert int16_t to Fp16T
     * @return  Return Fp16T result from iVal
     */
    Fp16T& operator = (const int16_t& iVal);
    /* *
     * @ingroup Fp16T math evaluation operator
     * @param [in] uiVal uint16_t object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert uint16_t to Fp16T
     * @return  Return Fp16T result from uiVal
     */
    Fp16T& operator = (const uint16_t& uiVal);
    /* *
     * @ingroup Fp16T math evaluation operator
     * @param [in] iVal int32_t object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert int32_t to Fp16T
     * @return  Return Fp16T result from iVal
     */
    Fp16T& operator = (const int32_t& iVal);
    /* *
     * @ingroup Fp16T math evaluation operator
     * @param [in] uiVal uint32_t object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert uint32_t to Fp16T
     * @return  Return Fp16T result from uiVal
     */
    Fp16T& operator = (const uint32_t& uiVal);
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to float/fp32
     * @return  Return float/fp32 value of Fp16T
     */
    operator float() const;
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to double/fp64
     * @return  Return double/fp64 value of Fp16T
     */
    operator double() const;
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to int8_t
     * @return  Return int8_t value of Fp16T
     */
    operator int8_t() const;
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to uint8_t
     * @return  Return uint8_t value of Fp16T
     */
    operator uint8_t() const;
    /* *
     * @ingroup Fp16T conversion
     * @brief   Override convert operator to convert Fp16T to int16_t
     * @return  Return int16_t value of Fp16T
     */
    operator int16_t() const;
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to uint16_t
     * @return  Return uint16_t value of Fp16T
     */
    operator uint16_t() const;
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to int32_t
     * @return  Return int32_t value of Fp16T
     */
    operator int32_t() const;
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to int64_t
     * @return  Return int64_t value of Fp16T
     */
    operator uint32_t() const;
    /* *
     * @ingroup Fp16T judgment method
     * @param [in] fp Fp16T object to be judgement
     * @brief   whether a Fp16T is inifinite
     * @return  Returns 1:+INF -1:-INF 0:not INF
     */
    int32_t IsInf() const;
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to float/fp32
     * @return  Return float/fp32 value of Fp16T
     */
    float ToFloat() const;
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to double/fp64
     * @return  Return double/fp64 value of Fp16T
     */
    double ToDouble() const;
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to int8_t
     * @return  Return int8_t value of Fp16T
     */
    int8_t ToInt8() const;
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to uint8_t
     * @return  Return uint8_t value of Fp16T
     */
    uint8_t ToUInt8() const;
    /* *
     * @ingroup Fp16T conversion
     * @brief   Convert Fp16T to int16_t
     * @return  Return int16_t value of Fp16T
     */
    int16_t ToInt16() const;
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to uint16_t
     * @return  Return uint16_t value of Fp16T
     */
    uint16_t ToUInt16() const;
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to int32_t
     * @return  Return int32_t value of Fp16T
     */
    int32_t ToInt32() const;
    /* *
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to int64_t
     * @return  Return int64_t value of Fp16T
     */
    uint32_t ToUInt32() const;
};

/**
 * @ingroup Fp16T public method
 * @param [in]     val signature is negative
 * @param [in|out] s   sign of Fp16T object
 * @param [in|out] e   exponent of Fp16T object
 * @param [in|out] m   mantissa of Fp16T object
 * @brief   Extract the sign, exponent and mantissa of a Fp16T object
 */
void ExtractFp16(const uint16_t& val, uint16_t& s, int16_t& e, uint16_t& m);
/**
 * @ingroup Fp16T public method
 * @param [in]     negative sign is negative
 * @param [in|out] man      mantissa to be reverse
 * @brief   Calculate a mantissa's complement (add ont to it's radix-minus-one
 * complement)
 * @return  Return complement of man
 */
template <typename T> void ReverseMan(bool negative, T& man)
{
    if (negative) {
        man = (~(man)) + 1;
    }
}
/**
 * @ingroup Fp16T public method
 * @param [in] ea exponent of one Fp16T/float number
 * @param [in] ma mantissa of one Fp16T/float number
 * @param [in] eb exponent of another Fp16T/float number
 * @param [in] mb mantissa of another Fp16T/float number
 * @brief   choose mantissa to be shift right whoes exponent is less than another
 * one
 * @return  Return mantissawhoes exponent is less than another one
 */
template <typename T> T MinMan(const int16_t& ea, T& ma, const int16_t& eb, T& mb)
{
    return (ea > eb) ? mb : ma;
}

/**
 * @ingroup Fp16T public method
 * @param [in] man   mantissa to be operate
 * @param [in] shift right shift bits
 * @brief   right shift a mantissa
 * @return  Return right-shift mantissa
 */
template <typename T> T RightShift(T man, int16_t shift)
{
    int32_t bits = sizeof(T) * 8; // one byte have 8 bits
    T mask = ((static_cast<T>(1u)) << (static_cast<uint32_t>(bits - 1)));
    for (int32_t i = 0; i < shift; i++) {
        man = ((man & mask) | (man >> 1));
    }
    return man;
}

/**
 * @ingroup Fp16T public method
 * @param [in] ea exponent of one temp Fp16T number
 * @param [in] ma mantissa of one temp Fp16T number
 * @param [in] eb exponent of another temp Fp16T number
 * @param [in] mb mantissa of another temp Fp16T number
 * @brief   Get mantissa sum of two temp Fp16T numbers, T support types:
 * uint16_t/uint32_t/uint64_t
 * @return  Return mantissa sum
 */
template <typename T> T GetManSum(int16_t ea, const T& ma, int16_t eb, const T& mb)
{
    T sum;
    if (ea != eb) {
        T mTmp;
        int16_t eTmp = static_cast<int16_t>(std::abs(ea - eb));
        if (ea > eb) {
            mTmp = mb;
            mTmp = RightShift(mTmp, eTmp);
            sum = ma + mTmp;
        } else {
            mTmp = ma;
            mTmp = RightShift(mTmp, eTmp);
            sum = mTmp + mb;
        }
    } else {
        sum = mb + ma;
    }
    return sum;
}

/**
 * @ingroup Fp16T public method
 * @param [in] bit0    whether the last preserved bit is 1 before round
 * @param [in] bit1    whether the abbreviation's highest bit is 1
 * @param [in] bitLeft whether the abbreviation's bits which not contain highest
 * bit grater than 0
 * @param [in] man     mantissa of a Fp16T or float number, support types:
 * uint16_t/uint32_t/uint64_t
 * @param [in] shift   abbreviation bits
 * @brief    Round Fp16T or float mantissa to nearest value
 * @return   Returns true if round 1,otherwise false;
 */
template <typename T> T ManRoundToNearest(bool bit0, bool bit1, bool bitLeft, T man, uint16_t shift = 0)
{
    man = ((bit1 && (bit0 || bitLeft)) ? 1 : 0) + (man >> shift);
    return man;
}

/**
 * @ingroup Fp16T public method
 * @param [in] man    mantissa of a float number, support types: uint16_t/uint32_t/uint64_t
 * @brief   Get bit length of a uint32_t number
 * @return  Return bit length of man
 */
template <typename T> int16_t GetManBitLength(T man)
{
    int16_t lenRet = 0;
    while (man) {
        lenRet++;
        man >>= 1;
    }
    return lenRet;
}

/**
 * \brief half datatype
 *
 * \details This structure implements the datatype for storing half-precision floating-point numbers.
 * The structure implements assignment operators and type conversions.
 * 16 bits are being used in total: 1 sign bit, 5 bits for the exponent, and the significand is
 * being stored in 10 bits.
 * The total precision is 11 bits. There are 15361 representable numbers within theinterval [0.0, 1.0],
 * endpoints included.
 * On average we have log10(2**11) â‰ˆ 3.311 decimal digits.
 */
} // namespace float16
using half = float16::Fp16T;
#endif // ASCENDC_FP16_H
