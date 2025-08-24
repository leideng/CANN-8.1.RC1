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

/*!
 * \file conv_config.h
 * \brief
 */

#ifndef CONV_CONFIG_H
#define CONV_CONFIG_H

#include "kernel_utils.h"
#include "conv_util.h"
#include "conv_framework_util.h"

using namespace AscendC;

namespace conv {

enum ConvCfgTypeID {
    CONV_ID_Unknown,
    CONV_ID_Normal,
    CONV_ID_Test,
    CONV_ID_END  // 由于编译器限制，最大为: 1024
};

enum class ConvFormat { ND = 0, NCHW, NHWC, NC1HWC0, FRACTAL_Z, NDC1HWC0, FRACTAL_Z_3D, NCDHW };

struct ConvParam {
    __aicore__ inline ConvParam(){};
    constexpr static int8_t outputOrder = -1;
    constexpr static int8_t l0pingpong = -1;
    constexpr static int8_t bl1bypass = -1;
    constexpr static int8_t groupConvType = -1;
};

CONV_DECLARE_DEFINE_MEMBER(ConvParam, outputOrder, int8_t, -1)
CONV_DECLARE_DEFINE_MEMBER(ConvParam, l0pingpong, int8_t, -1)
CONV_DECLARE_DEFINE_MEMBER(ConvParam, bl1bypass, int8_t, -1)
CONV_DECLARE_DEFINE_MEMBER(ConvParam, groupConvType, int8_t, -1)

template <typename T>
struct GetDstType {
    using Type = T;
};

template <>
struct GetDstType<float> {
    using Type = float;
};

template <>
struct GetDstType<half> {
    using Type = float;
};

template <>
struct GetDstType<bfloat16_t> {
    using Type = float;
};

template <>
struct GetDstType<int8_t> {
    using Type = int32_t;
};

template <TPosition POSITION, ConvFormat FORMAT, typename TYPE>
struct ConvType {
    constexpr static TPosition pos = POSITION;
    constexpr static ConvFormat format = FORMAT;
    using T = TYPE;
};

template <class INPUT_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class CONV_CFG>
struct ConvDataType {
    using ConvParam = CONV_CFG;
    using SrcT = typename INPUT_TYPE::T;
    using SrcAT = typename INPUT_TYPE::T;
    using SrcBT = typename WEIGHT_TYPE::T;
    using DstT = typename OUTPUT_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using L0cT = typename GetDstType<SrcT>::Type;

    constexpr static uint32_t configID = (uint32_t)ConvCfgTypeID::CONV_ID_Normal;
    constexpr static uint32_t implType = (uint32_t)ConvCfgTypeID::CONV_ID_Normal;
    constexpr static uint32_t intfType = (uint32_t)ConvCfgTypeID::CONV_ID_Normal;

    constexpr static auto formatA = INPUT_TYPE::format;
    constexpr static auto formatB = WEIGHT_TYPE::format;
    constexpr static auto formatC = OUTPUT_TYPE::format;
    constexpr static auto formatBias = BIAS_TYPE::format;

    constexpr static auto posA = INPUT_TYPE::pos;
    constexpr static auto posB = WEIGHT_TYPE::pos;
    constexpr static auto posC = OUTPUT_TYPE::pos;
    constexpr static auto posBias = BIAS_TYPE::pos;

    constexpr static bool isBias = true;

    using ContextData = struct _ {
        __aicore__ inline _()
        {}
    };
};

template <class ConvDataType>
struct ConvConfig : public ConvDataType {
public:
    __aicore__ inline ConvConfig()
    {}

    using ContextData = struct _ : public ConvDataType::ContextData {
        __aicore__ inline _()
        {}
        int test1 = 11;
    };
};

}  // namespace conv
#endif
