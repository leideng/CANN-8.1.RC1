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

/* !
 * \file kernel_operator_vec_createvecindex_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H
#include "kernel_tensor.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
constexpr float Fp32IdxIncrement = 64;
const half Fp16IdxIncrement = 128;
template <typename T> constexpr __aicore__ inline void CheckCreateVecIndexApi0SupportedType()
{
    static_assert(SupportType<T, int16_t, int32_t, half, float>(),
        "CreateVecIndex level-0 api only support int16_t/int32_t/half/float on current device");
}

template <typename T> constexpr __aicore__ inline void CheckCreateVecIndexApi2SupportedType()
{
    static_assert(SupportType<T, int16_t, int32_t, half, float>(),
        "CreateVecIndex level-2 api only support int16_t/int32_t/half/float/ on current device");
}

// VCI level-0 normal
template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> &dstLocal, const T firstValue, uint64_t mask,
    uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    CheckCreateVecIndexApi0SupportedType<T>();
}

template <typename T = int16_t>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<int16_t> &dstLocal, const int16_t firstValue, uint64_t mask,
    uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    __ubuf__ T *dstLocalAddr = (__ubuf__ T *)dstLocal.GetPhyAddr();
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    constexpr uint32_t blockCount = (uint32_t)(ONE_BLK_SIZE / sizeof(T));
    uint32_t sreg = (uint32_t)mask;
    uint32_t strideConfig1 = (((uint32_t)dstBlkStride) << 16);

    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        vci(vreg0, firstValue, INC_ORDER);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsstb(vreg0, dstLocalAddr + i * dstRepStride * blockCount, strideConfig1, preg);
            vadds(vreg0, vreg0, sregLower, preg, MODE_ZEROING);
        }
    }
}

template <typename T = half>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<half> &dstLocal, const half firstValue, uint64_t mask,
    uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    __ubuf__ T *dstLocalAddr = (__ubuf__ T *)dstLocal.GetPhyAddr();
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    constexpr uint32_t blockCount = (uint32_t)(ONE_BLK_SIZE / sizeof(T));
    uint32_t sreg = (uint32_t)mask;
    uint32_t strideConfig1 = (((uint32_t)dstBlkStride) << 16);

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        vci(vreg0, firstValue, INC_ORDER);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsstb(vreg0, dstLocalAddr + i * dstRepStride * blockCount, strideConfig1, preg);
            vadds(vreg0, vreg0, Fp16IdxIncrement, preg, MODE_ZEROING);
        }
    }
}

template <typename T = int32_t>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<int32_t> &dstLocal, const int32_t firstValue, uint64_t mask,
    uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    __ubuf__ T *dstLocalAddr = (__ubuf__ T *)dstLocal.GetPhyAddr();
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    constexpr uint32_t blockCount = (uint32_t)(ONE_BLK_SIZE / sizeof(T));
    uint32_t sreg = (uint32_t)mask;
    uint32_t strideConfig1 = (((uint32_t)dstBlkStride) << 16);

    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        vci(vreg0, firstValue, INC_ORDER);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsstb(vreg0, dstLocalAddr + i * dstRepStride * blockCount, strideConfig1, preg);
            vadds(vreg0, vreg0, sregLower, preg, MODE_ZEROING);
        }
    }
}

template <typename T = float>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<float> &dstLocal, const float firstValue, uint64_t mask,
    uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    __ubuf__ T *dstLocalAddr = (__ubuf__ T *)dstLocal.GetPhyAddr();
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    constexpr uint32_t blockCount = (uint32_t)(ONE_BLK_SIZE / sizeof(T));
    uint32_t sreg = (uint32_t)mask;
    uint32_t strideConfig1 = (((uint32_t)dstBlkStride) << 16);

    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        vci(vreg0, firstValue, INC_ORDER);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsstb(vreg0, dstLocalAddr + i * dstRepStride * blockCount, strideConfig1, preg);
            vadds(vreg0, vreg0, Fp32IdxIncrement, preg, MODE_ZEROING);
        }
    }
}

// VCI level-0 bitwise
template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> &dstLocal, const T firstValue, uint64_t mask[],
    uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    CheckCreateVecIndexApi0SupportedType<T>();
}

template <typename T = int16_t>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<int16_t> &dstLocal, const int16_t firstValue, uint64_t mask[],
    uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    SetVectorMask<T>(mask[1], mask[0]);

    __ubuf__ int16_t *dstLocalAddr = (__ubuf__ int16_t *)dstLocal.GetPhyAddr();
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    constexpr uint32_t blockCount = (uint32_t)(ONE_BLK_SIZE / sizeof(T));
    uint32_t strideConfig1 = (((uint32_t)dstBlkStride) << 16);

    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_bool preg = movp_b16();
        vci(vreg0, firstValue, INC_ORDER);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsstb(vreg0, dstLocalAddr + i * dstRepStride * blockCount, strideConfig1, preg);
            vadds(vreg0, vreg0, sregLower, preg, MODE_ZEROING);
        }
    }
}

template <typename T = int32_t>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<int32_t> &dstLocal, const int32_t firstValue, uint64_t mask[],
    uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    SetVectorMask<T>(mask[1], mask[0]);

    __ubuf__ int32_t *dstLocalAddr = (__ubuf__ int32_t *)dstLocal.GetPhyAddr();
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    constexpr uint32_t blockCount = (uint32_t)(ONE_BLK_SIZE / sizeof(T));
    uint32_t strideConfig1 = (((uint32_t)dstBlkStride) << 16);

    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_bool preg = movp_b32();
        vci(vreg0, firstValue, INC_ORDER);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsstb(vreg0, dstLocalAddr + i * dstRepStride * blockCount, strideConfig1, preg);
            vadds(vreg0, vreg0, sregLower, preg, MODE_ZEROING);
        }
    }
}

template <typename T = half>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<half> &dstLocal, const half firstValue, uint64_t mask[],
    uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    SetVectorMask<T>(mask[1], mask[0]);

    __ubuf__ half *dstLocalAddr = (__ubuf__ half *)dstLocal.GetPhyAddr();
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    constexpr uint32_t blockCount = (uint32_t)(ONE_BLK_SIZE / sizeof(T));
    uint32_t strideConfig1 = (((uint32_t)dstBlkStride) << 16);

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_bool preg = movp_b16();
        vci(vreg0, firstValue, INC_ORDER);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsstb(vreg0, dstLocalAddr + i * dstRepStride * blockCount, strideConfig1, preg);
            vadds(vreg0, vreg0, Fp16IdxIncrement, preg, MODE_ZEROING);
        }
    }
}

template <typename T = float>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<float> &dstLocal, const float firstValue, uint64_t mask[],
    uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    SetVectorMask<T>(mask[1], mask[0]);

    __ubuf__ float *dstLocalAddr = (__ubuf__ float *)dstLocal.GetPhyAddr();
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    constexpr uint32_t blockCount = (uint32_t)(ONE_BLK_SIZE / sizeof(T));
    uint32_t strideConfig1 = (((uint32_t)dstBlkStride) << 16);

    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_bool preg = movp_b32();
        vci(vreg0, firstValue, INC_ORDER);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsstb(vreg0, dstLocalAddr + i * dstRepStride * blockCount, strideConfig1, preg);
            vadds(vreg0, vreg0, Fp32IdxIncrement, preg, MODE_ZEROING);
        }
    }
}

// VCI level-2
template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> dstLocal, const T firstValue, uint32_t calCount)
{
    CheckCreateVecIndexApi2SupportedType<T>();
}

template <typename T = int16_t>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<int16_t> dstLocal, const int16_t firstValue, uint32_t calCount)
{
    __ubuf__ T *dstLocalAddr = (__ubuf__ T *)dstLocal.GetPhyAddr();
    uint32_t sreg = (uint32_t)calCount;
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    uint16_t repeatTimes = CeilDivision(calCount, sregLower);

    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_bool preg;
        vci(vreg0, firstValue, INC_ORDER);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vector_bool preg = plt_b16(sreg, POST_UPDATE);
            vsts(vreg0, dstLocalAddr, i * sregLower, NORM_B16, preg);
            vadds(vreg0, vreg0, sregLower, preg, MODE_ZEROING);
        }
    }
}

template <typename T = half>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<half> dstLocal, const half firstValue, uint32_t calCount)
{
    __ubuf__ T *dstLocalAddr = (__ubuf__ T *)dstLocal.GetPhyAddr();
    uint32_t sreg = (uint32_t)calCount;
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    uint16_t repeatTimes = CeilDivision(calCount, sregLower);

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_bool preg;
        vci(vreg0, firstValue, INC_ORDER);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vector_bool preg = plt_b16(sreg, POST_UPDATE);
            vsts(vreg0, dstLocalAddr, i * sregLower, NORM_B16, preg);
            vadds(vreg0, vreg0, Fp16IdxIncrement, preg, MODE_ZEROING);
        }
    }
}

template <typename T = int32_t>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<int32_t> dstLocal, const int32_t firstValue, uint32_t calCount)
{
    __ubuf__ T *dstLocalAddr = (__ubuf__ T *)dstLocal.GetPhyAddr();
    uint32_t sreg = (uint32_t)calCount;
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    uint16_t repeatTimes = CeilDivision(calCount, sregLower);

    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_bool preg;
        vci(vreg0, firstValue, INC_ORDER);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vector_bool preg = plt_b32(sreg, POST_UPDATE);
            vsts(vreg0, dstLocalAddr, i * sregLower, NORM_B32, preg);
            vadds(vreg0, vreg0, sregLower, preg, MODE_ZEROING);
        }
    }
}

template <typename T = float>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<float> dstLocal, const float firstValue, uint32_t calCount)
{
    __ubuf__ T *dstLocalAddr = (__ubuf__ T *)dstLocal.GetPhyAddr();
    uint32_t sreg = (uint32_t)calCount;
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    uint16_t repeatTimes = CeilDivision(calCount, sregLower);

    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_bool preg;
        vci(vreg0, firstValue, INC_ORDER);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vector_bool preg = plt_b32(sreg, POST_UPDATE);
            vsts(vreg0, dstLocalAddr, i * sregLower, NORM_B32, preg);
            vadds(vreg0, vreg0, Fp32IdxIncrement, preg, MODE_ZEROING);
        }
    }
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H