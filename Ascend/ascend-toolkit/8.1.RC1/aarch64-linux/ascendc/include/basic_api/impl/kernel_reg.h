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
 * \file kernel_reg.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_REG_IMPL_H
#define ASCENDC_KERNEL_REG_IMPL_H

#include "kernel_utils.h"
#include "kernel_struct_aipp.h"

namespace AscendC {
constexpr uint64_t MASK_PLACEHOLDER = 0;
constexpr uint64_t MASK_PLACEHOLDER_LIST[2] = {0, 0};

enum class MaskMode : uint8_t {
    NORMAL = 0,
    COUNTER
};

template <typename T, MaskMode mode>
__aicore__ static inline void SetVectorMaskImpl(const uint64_t maskHigh, const uint64_t maskLow)
{
    if ASCEND_IS_NOT_AIC {
        set_vector_mask(maskHigh, maskLow);
    }
}

template <typename T, MaskMode mode>
__aicore__ static inline void SetVectorMaskImpl(int32_t len)
{
    if constexpr (mode == MaskMode::COUNTER) {
        SetVectorMaskImpl<T, mode>(0, len);
        return;
    }
    AscendCUtils::SetMask<T>(len);
}

__aicore__ inline void ResetMaskImpl()
{
    if ASCEND_IS_NOT_AIC {
        set_vector_mask(FULL_MASK, FULL_MASK);
    }
}

template <pipe_t pipe> __aicore__ inline void PipeBarrierImpl()
{
#if defined(__DAV_M310__)
    return;
#endif

#if (__CCE_AICORE__ == 300)
    if constexpr (pipe == PIPE_S || pipe == PIPE_V) {
        return;
    }
#endif
#if (__CCE_AICORE__ == 220)
    if ASCEND_IS_AIC {
        if constexpr (pipe == PIPE_V) {
            return;
        }
    }
#endif
    pipe_barrier(pipe);
}

enum class CacheLine : uint64_t {
    SINGLE_CACHE_LINE = 0,
    ENTIRE_DATA_CACHE
};

enum class DcciDst : uint64_t {
    CACHELINE_ALL = 0,
    CACHELINE_UB,
    CACHELINE_OUT,
    CACHELINE_ATOMIC
};

#if (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 300)
template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DcciGMImpl(__gm__ T* dst)
{
    dcci(static_cast<__gm__ void *>(dst), static_cast<uint64_t>(entireType), static_cast<uint64_t>(dcciDst));
}

template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DcciUBImpl(__ubuf__ T* dst)
{
    dcci(static_cast<__ubuf__ void *>(dst), static_cast<uint64_t>(entireType), static_cast<uint64_t>(dcciDst));
}
#endif

#if (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 200) || (__CCE_AICORE__ == 300)
template <typename T, CacheLine entireType>
__aicore__ inline void DcciGMImpl(__gm__ T* dst)
{
    dcci(static_cast<__gm__ void *>(dst), static_cast<uint64_t>(entireType));
}
#endif

__aicore__ inline void SetMaskCountImpl()
{
    set_mask_count();
}

__aicore__ inline void SetMaskNormImpl()
{
    set_mask_norm();
}

__aicore__ inline void SetLreluMode(bool lreluMode)
{
    if (lreluMode) {
        set_ctrl(sbitset1(get_ctrl(), LEAKY_RELU_MODE_BIT));
    } else {
        set_ctrl(sbitset0(get_ctrl(), LEAKY_RELU_MODE_BIT));
    }
}

__aicore__ inline void SetHF32ModeImpl(bool hf32Mode)
{
    if (hf32Mode) {
        set_ctrl(sbitset1(get_ctrl(), HF32_MODE_BIT));
    } else {
        set_ctrl(sbitset0(get_ctrl(), HF32_MODE_BIT));
    }
}

__aicore__ inline void SetHF32TransModeImpl(bool hf32TransMode)
{
    if (hf32TransMode) {
        set_ctrl(sbitset1(get_ctrl(), HF32_TRANS_MODE_BIT));
    } else {
        set_ctrl(sbitset0(get_ctrl(), HF32_TRANS_MODE_BIT));
    }
}

__aicore__ inline void SetMMLayoutTransformImpl(bool mmLayoutMode)
{
    if (mmLayoutMode) {
        set_ctrl(sbitset1(get_ctrl(), MM_LAYOUT_MODE_BIT));
    } else {
        set_ctrl(sbitset0(get_ctrl(), MM_LAYOUT_MODE_BIT));
    }
}

template <bool castMode>
__aicore__ inline void SetCastOverflowModeImpl()
{
    if constexpr (castMode) {
        set_ctrl(sbitset1(get_ctrl(), CAST_MODE_BIT));
    } else {
        set_ctrl(sbitset0(get_ctrl(), CAST_MODE_BIT));
    }
}

#if (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 200) || (__CCE_AICORE__ == 300)
template <typename T>
__aicore__ inline void SetAippFunctionsImpl0(__gm__ T* src0)
{
    uint64_t aippConfig0 = reinterpret_cast<uint64_t>(src0) & 0xffffffffffff;

    set_aipp_spr_0(aippConfig0);
}

template <typename T, typename U>
__aicore__ inline void SetAippFunctionsImpl1(__gm__ T* src1, AippParams<U>& config)
{
    uint64_t aippConfig1 = reinterpret_cast<uint64_t>(src1) & 0xffffffffffff;

    if (config.cscParams.isEnableCsc) {
        aippConfig1 |= static_cast<uint64_t>(1) << AIPP_OFFSET_CSC_ENABLE;
    }

    set_aipp_spr_1(aippConfig1);
}

template <typename U>
__aicore__ inline void SetAippFunctionsImpl2(AippParams<U>& config)
{
    uint16_t cscMatrixR0C0 = GetScalarBitcodeValue(config.cscParams.cscMatrixR0C0);
    uint16_t cscMatrixR0C1 = GetScalarBitcodeValue(config.cscParams.cscMatrixR0C1);
    uint16_t cscMatrixR0C2 = GetScalarBitcodeValue(config.cscParams.cscMatrixR0C2);
    uint16_t cscMatrixR1C0 = GetScalarBitcodeValue(config.cscParams.cscMatrixR1C0);

    uint64_t aippConfig2 = static_cast<uint64_t>(cscMatrixR0C0);
    aippConfig2 |= static_cast<uint64_t>(cscMatrixR0C1) << AIPP_OFFSET_CH1;
    aippConfig2 |= static_cast<uint64_t>(cscMatrixR0C2) << AIPP_OFFSET_CH2;
    aippConfig2 |= static_cast<uint64_t>(cscMatrixR1C0) << AIPP_OFFSET_CH3;

    set_aipp_spr_2(aippConfig2);
}

template <typename U>
__aicore__ inline void SetAippFunctionsImpl3(AippParams<U>& config)
{
    uint16_t cscMatrixR1C1 = GetScalarBitcodeValue(config.cscParams.cscMatrixR1C1);
    uint16_t cscMatrixR1C2 = GetScalarBitcodeValue(config.cscParams.cscMatrixR1C2);
    uint16_t cscMatrixR2C0 = GetScalarBitcodeValue(config.cscParams.cscMatrixR2C0);
    uint16_t cscMatrixR2C1 = GetScalarBitcodeValue(config.cscParams.cscMatrixR2C1);

    uint64_t aippConfig3 = static_cast<uint64_t>(cscMatrixR1C1);
    aippConfig3 |= static_cast<uint64_t>(cscMatrixR1C2) << AIPP_OFFSET_CH1;
    aippConfig3 |= static_cast<uint64_t>(cscMatrixR2C0)  << AIPP_OFFSET_CH2;
    aippConfig3 |= static_cast<uint64_t>(cscMatrixR2C1) << AIPP_OFFSET_CH3;

    set_aipp_spr_3(aippConfig3);
}

template <typename U>
__aicore__ inline void SetAippFunctionsImpl4(AippParams<U>& config)
{
    uint16_t cscMatrixR2C2 = GetScalarBitcodeValue(config.cscParams.cscMatrixR2C2);
    uint8_t cscBiasOut0 = GetScalarBitcodeValue(config.cscParams.cscBiasOut0);
    uint8_t cscBiasOut1 = GetScalarBitcodeValue(config.cscParams.cscBiasOut1);
    uint8_t cscBiasOut2 = GetScalarBitcodeValue(config.cscParams.cscBiasOut2);
    uint8_t cscBiasIn0 = GetScalarBitcodeValue(config.cscParams.cscBiasIn0);
    uint8_t cscBiasIn1 = GetScalarBitcodeValue(config.cscParams.cscBiasIn1);
    uint8_t cscBiasIn2 = GetScalarBitcodeValue(config.cscParams.cscBiasIn2);

    uint64_t aippConfig4 = static_cast<uint64_t>(cscMatrixR2C2);
    aippConfig4 |= static_cast<uint64_t>(cscBiasOut0) << AIPP_OFFSET_CSC_OUT_CH0;
    aippConfig4 |= static_cast<uint64_t>(cscBiasOut1) << AIPP_OFFSET_CSC_OUT_CH1;
    aippConfig4 |= static_cast<uint64_t>(cscBiasOut2) << AIPP_OFFSET_CSC_OUT_CH2;
    aippConfig4 |= static_cast<uint64_t>(cscBiasIn0) << AIPP_OFFSET_CSC_IN_CH0;
    aippConfig4 |= static_cast<uint64_t>(cscBiasIn1) << AIPP_OFFSET_CSC_IN_CH1;
    aippConfig4 |= static_cast<uint64_t>(cscBiasIn2) << AIPP_OFFSET_CSC_IN_CH2;

    set_aipp_spr_4(aippConfig4);
}

template <typename U>
__aicore__ inline void SetAippFunctionsImpl5(AippParams<U>& config)
{
#if __CCE_AICORE__ == 300
    return;
#endif
    uint8_t dtcMeanCh0 = GetScalarBitcodeValue(config.dtcParams.dtcMeanCh0);
    uint8_t dtcMeanCh1 = GetScalarBitcodeValue(config.dtcParams.dtcMeanCh1);
    uint8_t dtcMeanCh2 = GetScalarBitcodeValue(config.dtcParams.dtcMeanCh2);

    uint64_t aippConfig5 = static_cast<uint64_t>(dtcMeanCh0);
    aippConfig5 |= static_cast<uint64_t>(dtcMeanCh1) << AIPP_OFFSET_CH1;
    aippConfig5 |= static_cast<uint64_t>(dtcMeanCh2) << AIPP_OFFSET_CH2;

    set_aipp_spr_5(aippConfig5);
}

template <typename U>
__aicore__ inline void SetAippFunctionsImpl6(AippParams<U>& config)
{
#if __CCE_AICORE__ == 300
    return;
#endif
    uint16_t dtcMinCh0 = GetScalarBitcodeValue(config.dtcParams.dtcMinCh0);
    uint16_t dtcMinCh1 = GetScalarBitcodeValue(config.dtcParams.dtcMinCh1);
    uint16_t dtcMinCh2 = GetScalarBitcodeValue(config.dtcParams.dtcMinCh2);

    uint64_t aippConfig6 = static_cast<uint64_t>(dtcMinCh0);
    aippConfig6 |= static_cast<uint64_t>(dtcMinCh1) << AIPP_OFFSET_CH1;
    aippConfig6 |= static_cast<uint64_t>(dtcMinCh2) << AIPP_OFFSET_CH2;

    set_aipp_spr_6(aippConfig6);
}

template <typename U>
__aicore__ inline void SetAippFunctionsImpl7(AippParams<U>& config)
{
#if __CCE_AICORE__ == 300
    return;
#endif
    uint16_t dtcVarCh0 = GetScalarBitcodeValue(config.dtcParams.dtcVarCh0);
    uint16_t dtcVarCh1 = GetScalarBitcodeValue(config.dtcParams.dtcVarCh1);
    uint16_t dtcVarCh2 = GetScalarBitcodeValue(config.dtcParams.dtcVarCh2);

    uint64_t aippConfig7 = static_cast<uint64_t>(dtcVarCh0);
    aippConfig7 |= static_cast<uint64_t>(dtcVarCh1) << AIPP_OFFSET_CH1;
    aippConfig7 |= static_cast<uint64_t>(dtcVarCh2) << AIPP_OFFSET_CH2;

    set_aipp_spr_7(aippConfig7);
}

template <typename U>
__aicore__ inline void SetAippFunctionsImpl8(AippParams<U>& config)
{
    uint64_t aippConfig8 = 0;
    if constexpr(IsSameType<U, int8_t>::value || IsSameType<U, uint8_t>::value) {
        uint8_t paddingValueCh0 = GetScalarBitcodeValue(config.paddingParams.paddingValueCh0);
        uint8_t paddingValueCh1 = GetScalarBitcodeValue(config.paddingParams.paddingValueCh1);
        uint8_t paddingValueCh2 = GetScalarBitcodeValue(config.paddingParams.paddingValueCh2);
        uint8_t paddingValueCh3 = GetScalarBitcodeValue(config.paddingParams.paddingValueCh3);

        aippConfig8 |= static_cast<uint64_t>(paddingValueCh0);
        aippConfig8 |= static_cast<uint64_t>(paddingValueCh1) << AIPP_OFFSET_CH1;
        aippConfig8 |= static_cast<uint64_t>(paddingValueCh2) << AIPP_OFFSET_CH2;
        aippConfig8 |= static_cast<uint64_t>(paddingValueCh3) << AIPP_OFFSET_CH3;
    } else {
        uint16_t paddingValueCh0 = GetScalarBitcodeValue(config.paddingParams.paddingValueCh0);
        uint16_t paddingValueCh1 = GetScalarBitcodeValue(config.paddingParams.paddingValueCh1);
        uint16_t paddingValueCh2 = GetScalarBitcodeValue(config.paddingParams.paddingValueCh2);
        uint16_t paddingValueCh3 = GetScalarBitcodeValue(config.paddingParams.paddingValueCh3);

        aippConfig8 |= static_cast<uint64_t>(paddingValueCh0);
        aippConfig8 |= static_cast<uint64_t>(paddingValueCh1) << AIPP_OFFSET_CH1;
        aippConfig8 |= static_cast<uint64_t>(paddingValueCh2) << AIPP_OFFSET_CH2;
        aippConfig8 |= static_cast<uint64_t>(paddingValueCh3) << AIPP_OFFSET_CH3;
    }

    set_aipp_spr_8(aippConfig8);
}

template <typename U>
__aicore__ inline void SetAippFunctionsImpl9(AippInputFormat format, AippParams<U>& config)
{
    uint64_t aippConfig9 = 0;

    if constexpr(IsSameType<U, int8_t>::value || IsSameType<U, uint8_t>::value) {
        uint8_t cPaddingValue = GetScalarBitcodeValue(config.cPaddingParams.cPaddingValue);
        aippConfig9 |= static_cast<uint64_t>(cPaddingValue);
    } else {
        uint16_t cPaddingValue = GetScalarBitcodeValue(config.cPaddingParams.cPaddingValue);
        aippConfig9 |= static_cast<uint64_t>(cPaddingValue);
    }

    if (config.swapParams.isSwapRB) {
        aippConfig9 |= static_cast<uint64_t>(1) << AIPP_OFFSET_SWAP_RB;
    }
    if (config.swapParams.isSwapUV) {
        aippConfig9 |= static_cast<uint64_t>(1) << AIPP_OFFSET_SWAP_UV;
    }
    if (config.swapParams.isSwapAX) {
        aippConfig9 |= static_cast<uint64_t>(1) << AIPP_OFFSET_SWAP_AX;
    }

    aippConfig9 |= (static_cast<uint64_t>(format) & 0x1f) << AIPP_OFFSET_FORMAT;

    if (config.singleLineParams.isSingleLineCopy) {
        aippConfig9 |= static_cast<uint64_t>(1) << AIPP_OFFSET_SINGLE_LINE;
    }

    aippConfig9 |= (static_cast<uint64_t>(config.paddingParams.paddingMode) & 0x3) << AIPP_OFFSET_PADDING_MODE;

#if __CCE_AICORE__ == 300
    aippConfig9 |= (static_cast<uint64_t>(config.dtcParams.dtcRoundMode) & 0x1) << AIPP_OFFSET_DTC_ROUND_MODE;
#endif

    aippConfig9 |= (static_cast<uint64_t>(config.cPaddingParams.cPaddingMode) & 0x1) << AIPP_OFFSET_CPADDING_MODE;

    set_aipp_spr_9(aippConfig9);
}

template <typename U>
__aicore__ inline void SetAippFunctionsImpl18(AippParams<U>& config)
{
#if __CCE_AICORE__ != 300
    return;
#endif
    float dtcVarCh0f = static_cast<float>(config.dtcParams.dtcVarCh0);
    float dtcVarCh1f = static_cast<float>(config.dtcParams.dtcVarCh1);
    uint32_t dtcVarCh0 = GetScalarBitcodeValue(dtcVarCh0f);
    uint32_t dtcVarCh1 = GetScalarBitcodeValue(dtcVarCh1f);

    uint64_t aippConfig18 = static_cast<uint64_t>(dtcVarCh0);
    aippConfig18 |= static_cast<uint64_t>(dtcVarCh1) << AIPP_OFFSET_DTC_CH1;

    set_aipp_spr_18(aippConfig18);
}

template <typename U>
__aicore__ inline void SetAippFunctionsImpl19(AippParams<U>& config)
{
#if __CCE_AICORE__ != 300
    return;
#endif
    float dtcVarCh2f = static_cast<float>(config.dtcParams.dtcVarCh2);
    uint32_t dtcVarCh2 = GetScalarBitcodeValue(dtcVarCh2f);
    uint64_t aippConfig19 = static_cast<uint64_t>(dtcVarCh2);
    set_aipp_spr_19(aippConfig19);
}

template <typename U>
__aicore__ inline void SetAippFunctionsImpl20(AippParams<U>& config)
{
#if __CCE_AICORE__ != 300
    return;
#endif
    float dtcMeanCh0f = static_cast<float>(config.dtcParams.dtcMeanCh0 * 1.0f);
    float dtcMeanCh1f = static_cast<float>(config.dtcParams.dtcMeanCh1 * 1.0f);

    uint32_t dtcMeanCh0 = GetScalarBitcodeValue(dtcMeanCh0f);
    uint32_t dtcMeanCh1 = GetScalarBitcodeValue(dtcMeanCh1f);

    uint64_t aippConfig20 = static_cast<uint64_t>(dtcMeanCh0);
    aippConfig20 |= static_cast<uint64_t>(dtcMeanCh1) << AIPP_OFFSET_DTC_CH1;

    set_aipp_spr_20(aippConfig20);
}

template <typename U>
__aicore__ inline void SetAippFunctionsImpl21(AippParams<U>& config)
{
#if __CCE_AICORE__ != 300
    return;
#endif
    float dtcMeanCh2f = static_cast<float>(config.dtcParams.dtcMeanCh2 * 1.0f);
    uint32_t dtcMeanCh2 = GetScalarBitcodeValue(dtcMeanCh2f);
    uint64_t aippConfig21 = static_cast<uint64_t>(dtcMeanCh2);
    set_aipp_spr_21(aippConfig21);
}

template <typename T, typename U>
__aicore__ inline void SetAippFunctionsImpl(__gm__ T* src0, __gm__ T* src1,
    AippInputFormat format, AippParams<U>& config)
{
#if __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
        return;
    }
#endif // __CCE_AICORE__ == 220
#if __CCE_AICORE__ == 300
    SetAippFunctionsImpl0<T>(src0);
    SetAippFunctionsImpl1<T, U>(src1, config);
    SetAippFunctionsImpl2<U>(config);
    SetAippFunctionsImpl3<U>(config);
    SetAippFunctionsImpl4<U>(config);
    SetAippFunctionsImpl8<U>(config);
    SetAippFunctionsImpl9<U>(format, config);
    SetAippFunctionsImpl18<U>(config);
    SetAippFunctionsImpl19<U>(config);
    SetAippFunctionsImpl20<U>(config);
    SetAippFunctionsImpl21<U>(config);
#else
    SetAippFunctionsImpl0<T>(src0);
    SetAippFunctionsImpl1<T, U>(src1, config);
    SetAippFunctionsImpl2<U>(config);
    SetAippFunctionsImpl3<U>(config);
    SetAippFunctionsImpl4<U>(config);
    SetAippFunctionsImpl5<U>(config);
    SetAippFunctionsImpl6<U>(config);
    SetAippFunctionsImpl7<U>(config);
    SetAippFunctionsImpl8<U>(config);
    SetAippFunctionsImpl9<U>(format, config);
#endif // __CCE_AICORE__ == 300
}

template <typename T, typename U>
__aicore__ inline void SetAippFunctionsImpl(__gm__ T* src0, AippInputFormat format, AippParams<U> config)
{
#if __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
        return;
    }
#endif // __CCE_AICORE__ == 220
    SetAippFunctionsImpl(src0, reinterpret_cast<__gm__ T*>(0), format, config);
}
#endif // (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 200) || (__CCE_AICORE__ == 300)

} // namespace AscendC
#endif // ASCENDC_KERNEL_REG_IMPL_H
