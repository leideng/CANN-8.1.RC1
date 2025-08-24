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
 * \file kernel_scalar.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_SCALAR_IMPL_H
#define ASCENDC_KERNEL_SCALAR_IMPL_H

namespace AscendC {
template <int countValue>
__aicore__ inline int64_t ScalarGetCountOfValueImpl(uint64_t valueIn)
{
    if constexpr (countValue == 1) {
        return bcnt1(valueIn);
    } else if constexpr (countValue == 0) {
        return bcnt0(valueIn);
    } else {
        static_assert(((countValue == 0) || (countValue == 1)) && "countValue must be 1 or 0");
        return 0;
    }
}

__aicore__ inline int64_t ScalarCountLeadingZeroImpl(uint64_t valueIn)
{
    return clz(valueIn);
}

__aicore__ inline int64_t CountBitsCntSameAsSignBitImpl(int64_t valueIn)
{
    return sflbits(valueIn);
}

template <int countValue>
__aicore__ inline int64_t ScalarGetSFFValueImpl(uint64_t valueIn)
{
    if constexpr (countValue == 1) {
        return sff1(valueIn);
    } else if constexpr (countValue == 0) {
        return sff0(valueIn);
    } else {
        static_assert(((countValue == 0) || (countValue == 1)) && "countValue must be 1 or 0");
        return 0;
    }
}

template <RoundMode roundMode>
__aicore__ inline half ScalarCastF322F16Impl(float valueIn)
{
    switch (roundMode) {
        case RoundMode::CAST_ODD:
            return conv_f322f16o(valueIn);
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            return 0;
    }
}

template <RoundMode roundMode>
__aicore__ inline int32_t ScalarCastF322S32Impl(float valueIn)
{
    switch (roundMode) {
        case RoundMode::CAST_ROUND:
            return conv_f322s32a(valueIn);
        case RoundMode::CAST_CEIL:
            return conv_f322s32c(valueIn);
        case RoundMode::CAST_FLOOR:
            return conv_f322s32f(valueIn);
        case RoundMode::CAST_RINT:
            return conv_f322s32r(valueIn);
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            return 0;
    }
}

template <typename srcT, typename dstT, RoundMode roundMode>
__aicore__ inline dstT ScalarCastImpl(srcT valueIn)
{
#if (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 200)
    if constexpr (std::is_same<dstT, half>::value) {
        return ScalarCastF322F16Impl<roundMode>(valueIn);
    } else if constexpr (std::is_same<dstT, int32_t>::value) {
        return ScalarCastF322S32Impl<roundMode>(valueIn);
    } else {
        static_assert(((sizeof(dstT) == sizeof(half)) || (sizeof(dstT) == sizeof(int32_t))),
            "dstT only support half or int32_t");
        return 0;
    }
#else
    ASCENDC_ASSERT((false), "ScalarCast is not supported on current device");
    return 0;
#endif
}
} // namespace AscendC
#endif // ASCENDC_KERNEL_SCALAR_IMPL_H