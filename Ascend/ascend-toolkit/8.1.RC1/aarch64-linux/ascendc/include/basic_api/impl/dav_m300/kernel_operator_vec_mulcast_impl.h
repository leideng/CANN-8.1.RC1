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
 * \file kernel_operator_vec_mulcast_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_MULCAST_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_MULCAST_IMPL_H
#include "kernel_tensor.h"
#include "kernel_struct_binary.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
template <typename T, typename U>
__aicore__ inline void MulCastCalc(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = int8_t, typename U = half>
__aicore__ inline void MulCastCalc(const LocalTensor<int8_t> &dstLocal, const LocalTensor<half> &src0Local,
    const LocalTensor<half> &src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<U, half>(), "MulCast level-0 api only support half on current device");
    static_assert(SupportType<T, int8_t>(), "MulCast level-0 api only support int8_t/uint8_t on current device");

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        vector_s8 vreg3;
        uint32_t sregLower = static_cast<uint32_t>(mask);
        uint32_t sregUpper = static_cast<uint32_t>(mask);
        vector_bool pregLower = plt_b8(sregLower, POST_UPDATE);
        vector_bool preg = plt_b16(sregUpper, POST_UPDATE);
        uint32_t strideConfig0 = (static_cast<uint32_t>(repeatParams.src0BlkStride) << 16);
        uint32_t strideConfig1 = (static_cast<uint32_t>(repeatParams.src1BlkStride) << 16);
        uint32_t strideConfig2 = (static_cast<uint32_t>(repeatParams.dstBlkStride) << 16);
        uint32_t strideOffset0 = static_cast<uint32_t>(repeatParams.src0RepStride * 256 / 16);
        uint32_t strideOffset1 = static_cast<uint32_t>(repeatParams.src1RepStride * 256 / 16);
        uint32_t strideOffset2 = static_cast<uint32_t>(repeatParams.dstRepStride * 256 / 8);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            mem_bar(VST_VLD);
            vsldb(vreg0, (__ubuf__ half*)src0Local.GetPhyAddr() + i * strideOffset0, strideConfig0, preg);
            vsldb(vreg1, (__ubuf__ half*)src1Local.GetPhyAddr() + i * strideOffset1, strideConfig1, preg);
            vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vcvt(vreg3, vreg2, preg, ROUND_R, RS_ENABLE, PART_EVEN);
            vpack((vector_u8&)vreg3, (vector_u16&)vreg3, LOWER, MODE_ZEROING);
            vsstb(vreg3, (__ubuf__ int8_t*)dstLocal.GetPhyAddr() + i * strideOffset2, strideConfig2, pregLower);
        }
    }
}

template <typename T = uint8_t, typename U = half>
__aicore__ inline void MulCastCalc(const LocalTensor<uint8_t> &dstLocal, const LocalTensor<half> &src0Local,
    const LocalTensor<half> &src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<U, half>(), "MulCast level-0 api only support half on current device");
    static_assert(SupportType<T, uint8_t>(), "MulCast level-0 api only support int8_t/uint8_t on current device");

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        vector_u8 vreg3;
        uint32_t sregLower = static_cast<uint32_t>(mask);
        uint32_t sregUpper = static_cast<uint32_t>(mask);
        vector_bool pregLower = plt_b8(sregLower, POST_UPDATE);
        vector_bool preg = plt_b16(sregUpper, POST_UPDATE);
        uint32_t strideConfig0 = (static_cast<uint32_t>(repeatParams.src0BlkStride) << 16);
        uint32_t strideConfig1 = (static_cast<uint32_t>(repeatParams.src1BlkStride) << 16);
        uint32_t strideConfig2 = (static_cast<uint32_t>(repeatParams.dstBlkStride) << 16);
        uint32_t strideOffset0 = static_cast<uint32_t>(repeatParams.src0RepStride * 256 / 16);
        uint32_t strideOffset1 = static_cast<uint32_t>(repeatParams.src1RepStride * 256 / 16);
        uint32_t strideOffset2 = static_cast<uint32_t>(repeatParams.dstRepStride * 256 / 8);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            mem_bar(VST_VLD);
            vsldb(vreg0, (__ubuf__ half*)src0Local.GetPhyAddr() + i * strideOffset0, strideConfig0, preg);
            vsldb(vreg1, (__ubuf__ half*)src1Local.GetPhyAddr() + i * strideOffset1, strideConfig1, preg);
            vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vcvt(vreg3, vreg2, preg, ROUND_R, RS_ENABLE, PART_EVEN);
            vpack((vector_u8&)vreg3, (vector_u16&)vreg3, LOWER, MODE_ZEROING);
            vsstb(vreg3, (__ubuf__ uint8_t*)dstLocal.GetPhyAddr() + i * strideOffset2, strideConfig2, pregLower);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void MulCastCalc(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = int8_t, typename U = half>
__aicore__ inline void MulCastCalc(const LocalTensor<int8_t> &dstLocal, const LocalTensor<half> &src0Local,
    const LocalTensor<half> &src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<U, half>(), "MulCast level-0 api only support half on current device");
    static_assert(SupportType<T, int8_t>(), "MulCast level-0 api only support int8_t/uint8_t on current device");

    SetVectorMask<U>(mask[1], mask[0]);

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        vector_s8 vreg3;
        vector_bool pregLower;
        vector_bool preg;
        preg = movp_b16();
        ppack(pregLower, preg, LOWER);
        uint32_t strideConfig0 = (static_cast<uint32_t>(repeatParams.src0BlkStride) << 16);
        uint32_t strideConfig1 = (static_cast<uint32_t>(repeatParams.src1BlkStride) << 16);
        uint32_t strideConfig2 = (static_cast<uint32_t>(repeatParams.dstBlkStride) << 16);
        uint32_t strideOffset0 = static_cast<uint32_t>(repeatParams.src0RepStride * 256 / 16);
        uint32_t strideOffset1 = static_cast<uint32_t>(repeatParams.src1RepStride * 256 / 16);
        uint32_t strideOffset2 = static_cast<uint32_t>(repeatParams.dstRepStride * 256 / 8);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            mem_bar(VST_VLD);
            vsldb(vreg0, (__ubuf__ half*)src0Local.GetPhyAddr() + i * strideOffset0, strideConfig0, preg);
            vsldb(vreg1, (__ubuf__ half*)src1Local.GetPhyAddr() + i * strideOffset1, strideConfig1, preg);
            vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vcvt(vreg3, vreg2, preg, ROUND_R, RS_ENABLE, PART_EVEN);
            vpack((vector_u8&)vreg3, (vector_u16&)vreg3, LOWER, MODE_ZEROING);
            vsstb(vreg3, (__ubuf__ int8_t*)dstLocal.GetPhyAddr() + i * strideOffset2, strideConfig2, pregLower);
        }
    }
}

template <typename T = uint8_t, typename U = half>
__aicore__ inline void MulCastCalc(const LocalTensor<uint8_t> &dstLocal, const LocalTensor<half> &src0Local,
    const LocalTensor<half> &src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<U, half>(), "MulCast level-0 api only support half on current device");
    static_assert(SupportType<T, uint8_t>(), "MulCast level-0 api only support int8_t/uint8_t on current device");

    SetVectorMask<U>(mask[1], mask[0]);

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        vector_u8 vreg3;
        vector_bool pregLower;
        vector_bool preg;
        preg = movp_b16();
        ppack(pregLower, preg, LOWER);
        uint32_t strideConfig0 = (static_cast<uint32_t>(repeatParams.src0BlkStride) << 16);
        uint32_t strideConfig1 = (static_cast<uint32_t>(repeatParams.src1BlkStride) << 16);
        uint32_t strideConfig2 = (static_cast<uint32_t>(repeatParams.dstBlkStride) << 16);
        uint32_t strideOffset0 = static_cast<uint32_t>(repeatParams.src0RepStride * 256 / 16);
        uint32_t strideOffset1 = static_cast<uint32_t>(repeatParams.src1RepStride * 256 / 16);
        uint32_t strideOffset2 = static_cast<uint32_t>(repeatParams.dstRepStride * 256 / 8);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            mem_bar(VST_VLD);
            vsldb(vreg0, (__ubuf__ half*)src0Local.GetPhyAddr() + i * strideOffset0, strideConfig0, preg);
            vsldb(vreg1, (__ubuf__ half*)src1Local.GetPhyAddr() + i * strideOffset1, strideConfig1, preg);
            vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vcvt(vreg3, vreg2, preg, ROUND_R, RS_ENABLE, PART_EVEN);
            vpack((vector_u8&)vreg3, (vector_u16&)vreg3, LOWER, MODE_ZEROING);
            vsstb(vreg3, (__ubuf__ uint8_t*)dstLocal.GetPhyAddr() + i * strideOffset2, strideConfig2, pregLower);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void MulCastCalc(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint32_t calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = int8_t, typename U = half>
__aicore__ inline void MulCastCalc(const LocalTensor<int8_t> &dstLocal, const LocalTensor<half> &src0Local,
    const LocalTensor<half> &src1Local, uint32_t calCount)
{
    static_assert(SupportType<T, int8_t>(), "MulCast level-2 api only support int8_t/uint8_t on current device");
    static_assert(SupportType<U, half>(), "MulCast level-2 api only support half on current device");

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        vector_s8 vreg3;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = static_cast<uint32_t>(128);
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, (__ubuf__ half*)src0Local.GetPhyAddr(), i * sregLower, NORM);
            vlds(vreg1, (__ubuf__ half*)src1Local.GetPhyAddr(), i * sregLower, NORM);
            vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vcvt(vreg3, vreg2, preg, ROUND_R, RS_ENABLE, PART_EVEN);
            vsts(vreg3, (__ubuf__ int8_t*)dstLocal.GetPhyAddr(), i * sregLower, PK_B16, preg);
        }
    }
}

template <typename T = uint8_t, typename U = half>
__aicore__ inline void MulCastCalc(const LocalTensor<uint8_t> &dstLocal, const LocalTensor<half> &src0Local,
    const LocalTensor<half> &src1Local, uint32_t calCount)
{
    static_assert(SupportType<T, uint8_t>(), "MulCast level-2 api only support int8_t/uint8_t on current device");
    static_assert(SupportType<U, half>(), "MulCast level-2 api only support half on current device");

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        vector_u8 vreg3;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = static_cast<uint32_t>(128);
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, (__ubuf__ half*)src0Local.GetPhyAddr(), i * sregLower, NORM);
            vlds(vreg1, (__ubuf__ half*)src1Local.GetPhyAddr(), i * sregLower, NORM);
            vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vcvt(vreg3, vreg2, preg, ROUND_R, RS_ENABLE, PART_EVEN);
            vsts(vreg3, (__ubuf__ uint8_t*)dstLocal.GetPhyAddr(), i * sregLower, PK_B16, preg);
        }
    }
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_MULCAST_IMPL_H