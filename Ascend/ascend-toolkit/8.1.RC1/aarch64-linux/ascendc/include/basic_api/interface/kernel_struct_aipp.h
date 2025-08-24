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
 * \file kernel_struct_aipp.h
 * \brief
 */
#ifndef ASCENDC_MODULE_STRUCT_AIPP_H
#define ASCENDC_MODULE_STRUCT_AIPP_H

namespace AscendC {
enum class AippInputFormat : uint8_t {
    YUV420SP_U8 = 0,
    XRGB8888_U8 = 1,
    RGB888_U8 = 4,
    YUV400_U8 = 9,
};

template <typename U>
struct AippPaddingParams {
    uint32_t paddingMode{ 0 };
    U paddingValueCh0{ 0 };
    U paddingValueCh1{ 0 };
    U paddingValueCh2{ 0 };
    U paddingValueCh3{ 0 };
};

struct AippSwapParams {
    bool isSwapRB{ false };
    bool isSwapUV{ false };
    bool isSwapAX{ false };
};

struct AippSingleLineParams {
    bool isSingleLineCopy{ false };
};

struct AippDataTypeConvParams {
    uint8_t dtcMeanCh0{ 0 };
    uint8_t dtcMeanCh1{ 0 };
    uint8_t dtcMeanCh2{ 0 };
    half dtcMinCh0{ 0 };
    half dtcMinCh1{ 0 };
    half dtcMinCh2{ 0 };
    half dtcVarCh0{ 1.0 };
    half dtcVarCh1{ 1.0 };
    half dtcVarCh2{ 1.0 };
    uint32_t dtcRoundMode{ 0 };
};

template <typename U>
struct AippChannelPaddingParams {
    uint32_t cPaddingMode;
    U cPaddingValue;
};

struct AippColorSpaceConvParams {
    bool isEnableCsc{ false };
    int16_t cscMatrixR0C0{ 0 };
    int16_t cscMatrixR0C1{ 0 };
    int16_t cscMatrixR0C2{ 0 };
    int16_t cscMatrixR1C0{ 0 };
    int16_t cscMatrixR1C1{ 0 };
    int16_t cscMatrixR1C2{ 0 };
    int16_t cscMatrixR2C0{ 0 };
    int16_t cscMatrixR2C1{ 0 };
    int16_t cscMatrixR2C2{ 0 };
    uint8_t cscBiasIn0{ 0 };
    uint8_t cscBiasIn1{ 0 };
    uint8_t cscBiasIn2{ 0 };
    uint8_t cscBiasOut0{ 0 };
    uint8_t cscBiasOut1{ 0 };
    uint8_t cscBiasOut2{ 0 };
};

#if (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 200) || (__CCE_AICORE__ == 300)
template <typename U>
struct AippParams {
    AippPaddingParams<U> paddingParams;
    AippSwapParams swapParams;
    AippSingleLineParams singleLineParams;
    AippDataTypeConvParams dtcParams;
    AippChannelPaddingParams<U> cPaddingParams;
    AippColorSpaceConvParams cscParams;
};
#endif // (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 200) || (__CCE_AICORE__ == 300)
} // namespace AscendC
#endif // ASCENDC_MODULE_STRUCT_AIPP_H