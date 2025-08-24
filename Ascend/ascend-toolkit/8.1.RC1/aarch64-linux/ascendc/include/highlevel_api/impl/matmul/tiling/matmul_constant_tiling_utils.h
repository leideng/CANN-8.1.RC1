/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_constant_tiling_utils.h
 * \brief
 */
#ifndef IMPL_MATMUL_TILING_MATMUL_CONSTANT_TILING_UTILS_H
#define IMPL_MATMUL_TILING_MATMUL_CONSTANT_TILING_UTILS_H

#include "matmul_constant_tiling_struct.h"

namespace AscendC {
namespace Impl {
constexpr int32_t C0_BYTE_SIZE = 32;
constexpr int32_t HW_C0 = 16;
constexpr int32_t DB_ON = 2;
constexpr int32_t DB_OFF = 1;
constexpr int32_t MIN_MTE1_LOAD = 32;
constexpr int32_t OUTER_STEP = 2;
#if __CCE_AICORE__ < 220
constexpr int32_t L1_SIZE = 1024 * 1024;
#elif __CCE_AICORE__ == 300
constexpr int32_t L1_SIZE = 1024 * 1024;
#elif __CCE_AICORE__ == 310
constexpr int32_t L1_SIZE = 512 * 1024;
#else
constexpr int32_t L1_SIZE = 512 * 1024;
#endif
}

enum class L1TilingType : uint8_t {
    KAL1_16,
    KBL1_16,
    M_AL1,
    N_BL1
};

struct L1Status {
    int32_t kAL1;
    int32_t kBL1;
    int32_t mAL1;
    int32_t nBL1;
    int32_t dbAL1;
    int32_t dbBL1;
    int32_t loadSize;
};

template <typename T>
__aicore__ constexpr int32_t GetReduceC0Size()
{
    return Impl::C0_BYTE_SIZE / GetBitSize<T>() * ONE_BYTE_BIT_SIZE;
}

__aicore__ constexpr int32_t GetML0(const MatmulConfig &mmCFG)
{
    return CeilNoLog<int32_t>(mmCFG.basicM, Impl::HW_C0);
}

__aicore__ constexpr int32_t GetNL0(const MatmulConfig &mmCFG)
{
    return CeilNoLog<int32_t>(mmCFG.basicN, Impl::HW_C0);
}

template <typename A_TYPE>
__aicore__ constexpr int32_t GetKL0(const MatmulConfig &mmCFG)
{
    using SrcAT = typename A_TYPE::T;
    return CeilNoLog<int32_t>(mmCFG.basicK, GetReduceC0Size<SrcAT>());
}

template <typename A_TYPE>
__aicore__ constexpr int32_t GetMTE1Loop(const MatmulConfig &mmCFG)
{
    int32_t nL0 = GetNL0(mmCFG);
    int32_t mL0 = GetML0(mmCFG);
    int32_t kL0 = GetKL0<A_TYPE>(mmCFG);
    return Impl::MIN_MTE1_LOAD / ((nL0 == 1 ? 1 : kL0) + (kL0 == 1 ? 1 : mL0));
}

__aicore__ constexpr int32_t GetMaxMAL1(const MatmulConfig &mmCFG)
{
    int32_t m = CeilNoLog<int32_t>(mmCFG.singleCoreM, Impl::HW_C0);
    int32_t mL0 = GetML0(mmCFG);
    return CeilNoLog<int32_t>(m, mL0);
}

__aicore__ constexpr int32_t GetMaxNBL1(const MatmulConfig &mmCFG)
{
    int32_t n = CeilNoLog<int32_t>(mmCFG.singleCoreN, Impl::HW_C0);
    int32_t nL0 = GetNL0(mmCFG);
    return CeilNoLog<int32_t>(n, nL0);
}

template <typename A_TYPE>
__aicore__ constexpr int32_t GetMaxKAL1(const MatmulConfig &mmCFG)
{
    int32_t mL0 = GetML0(mmCFG);
    int32_t kL0 = GetKL0<A_TYPE>(mmCFG);
    int32_t maxAL1 = ((Impl::MIN_MTE1_LOAD + mL0 - 1) / mL0 + kL0 - 1) / kL0;
    return MaxValue<int32_t>(maxAL1, GetMTE1Loop<A_TYPE>(mmCFG));
}

template <typename A_TYPE>
__aicore__ constexpr int32_t GetMaxKBL1(const MatmulConfig &mmCFG)
{
    int32_t nL0 = GetNL0(mmCFG);
    int32_t kL0 = GetKL0<A_TYPE>(mmCFG);
    int32_t maxBL1 = ((Impl::MIN_MTE1_LOAD + nL0 - 1) / nL0 + kL0 - 1) / kL0;
    return MaxValue<int32_t>(maxBL1, GetMTE1Loop<A_TYPE>(mmCFG));
}

template <typename A_TYPE>
__aicore__ constexpr int32_t GetKAAlignValue()
{
    using SrcAT = typename A_TYPE::T;
    if constexpr (sizeof(SrcAT) == sizeof(float)) {
        // when in FP32 mode, k_a must be an even number if k-alignment is needed. So make ka_align_value as 2.
        return A_TYPE::isTrans ? 2 : 1;
    }
    return 1;
}

template <typename A_TYPE, typename B_TYPE>
__aicore__ constexpr int32_t GetKBAlignValue()
{
    using SrcBT = typename B_TYPE::T;
    if constexpr (sizeof(SrcBT) == sizeof(float)) {
        // Same as previous one, make kb_align_value as 2 when k-alignment is needed
        return (A_TYPE::isTrans || !B_TYPE::isTrans) ? 2 : 1;
    }
    return 1;
}

template <typename BIAS_TYPE>
__aicore__ constexpr int32_t GetChannelWise(const MatmulConfig &mmCFG)
{
    using BiasT = typename BIAS_TYPE::T;
    if (mmCFG.enableSetBias) {
        return sizeof(BiasT) == sizeof(float) ? 2 : 1;
    } else {
        return 0;
    }
}

template <typename BIAS_TYPE>
__aicore__ constexpr int32_t GetBiasL1Size(const L1Status &l1Status, const MatmulConfig &mmCFG)
{
    int32_t biasSize = 0;
    if (mmCFG.enableSetBias) {
        if constexpr (PhyPosIsL1(BIAS_TYPE::pos)) {
            biasSize = 0;
        } else {
            int32_t channelWiseSize = GetChannelWise<BIAS_TYPE>(mmCFG) *
                l1Status.dbBL1 * GetTypeSize<typename BIAS_TYPE::T>();
            biasSize = l1Status.nBL1 * mmCFG.basicN * channelWiseSize;
        }
    }
    return biasSize;
}

__aicore__ constexpr int32_t GetDeQuantSize(const L1Status &l1Status, const MatmulConfig &mmCFG)
{
    int32_t dequantSize = 0;
    if (mmCFG.enableQuantVector) {
        dequantSize = l1Status.nBL1 * mmCFG.basicN * sizeof(uint64_t);
    }
    return dequantSize;
}

template <typename A_TYPE>
__aicore__ constexpr int32_t GetAL1Size(const L1Status &l1Status, const MatmulConfig &mmCFG)
{
    using SrcAT = typename A_TYPE::T;
    int32_t curA1Size = 0;
    int32_t kL0 = GetKL0<A_TYPE>(mmCFG);
    if constexpr (PhyPosIsL1(A_TYPE::pos)) {
        curA1Size = 0;
    } else {
        curA1Size = l1Status.mAL1 * mmCFG.basicM * CeilNoLog<int32_t>(l1Status.kAL1, kL0) * mmCFG.basicK *
            l1Status.dbAL1 * GetBitSize<SrcAT>() / ONE_BYTE_BIT_SIZE;
    }
    return curA1Size;
}

template <typename A_TYPE, typename B_TYPE>
__aicore__ constexpr int32_t GetBL1Size(const L1Status &l1Status, const MatmulConfig &mmCFG)
{
    int32_t curB1Size = 0;
    // B may different with A
    int32_t kL0 = GetKL0<A_TYPE>(mmCFG);
    if constexpr (PhyPosIsL1(B_TYPE::pos)) {
        curB1Size = 0;
    } else {
        curB1Size = l1Status.nBL1 * CeilNoLog<int32_t>(l1Status.kBL1, kL0) * l1Status.dbBL1 * mmCFG.basicN *
            mmCFG.basicK * GetBitSize<typename B_TYPE::T>() / ONE_BYTE_BIT_SIZE;
    }
    return curB1Size;
}

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
__aicore__ constexpr int32_t GetL1Size(const L1Status &l1Status, const MatmulConfig &mmCFG)
{
    int32_t curA1Size = GetAL1Size<A_TYPE>(l1Status, mmCFG);
    int32_t curB1Size = GetBL1Size<A_TYPE, B_TYPE>(l1Status, mmCFG);
    int32_t biasSize = GetBiasL1Size<BIAS_TYPE>(l1Status, mmCFG);
    int32_t dequantSize = GetDeQuantSize(l1Status, mmCFG);
    // overflow will generate complier error
    return curA1Size + curB1Size + biasSize + dequantSize;
}

template <typename BIAS_TYPE>
__aicore__ constexpr int32_t CalcL1MaxLen(int32_t l1LeftSize, const L1Status &l1Status,
    const MatmulConfig &mmCFG, int32_t alignValue, L1TilingType type)
{
    int32_t maxLen = 1;
    switch (type) {
        case L1TilingType::KAL1_16:
            maxLen = l1LeftSize / (l1Status.mAL1 * mmCFG.basicM * l1Status.dbAL1 * Impl::C0_BYTE_SIZE);
            maxLen = AlignDown<int32_t>(maxLen, alignValue);
            break;
        case L1TilingType::KBL1_16:
            maxLen = l1LeftSize / (l1Status.nBL1 * mmCFG.basicN * l1Status.dbBL1 * Impl::C0_BYTE_SIZE);
            maxLen = AlignDown<int32_t>(maxLen, alignValue);
            break;
        case L1TilingType::M_AL1:
            maxLen = l1LeftSize / (Align<int32_t>(l1Status.kAL1, alignValue) * mmCFG.basicM * l1Status.dbAL1 * Impl::C0_BYTE_SIZE);
            break;
        case L1TilingType::N_BL1:
            maxLen = l1LeftSize / (Align<int32_t>(l1Status.kBL1, alignValue) * mmCFG.basicN * l1Status.dbBL1 * Impl::C0_BYTE_SIZE +
                GetChannelWise<BIAS_TYPE>(mmCFG) * mmCFG.basicN * Impl::C0_BYTE_SIZE);
            break;
    }
    return maxLen;
}

__aicore__ constexpr int32_t GetNearestFactor(int32_t base, int32_t factor)
{
    int res = factor;
    while ((res > base) || (res > 0 && base % res != 0)) {
        res--;
    }
    return res;
}

template <typename A_TYPE, typename B_TYPE>
__aicore__ constexpr int32_t GetKMaxAxis(const MatmulConfig &mmCFG)
{
    int32_t kMaxAxis = 0;
    if constexpr (!A_TYPE::isTrans && !B_TYPE::isTrans) {
        kMaxAxis = 1;
    }
    if constexpr (A_TYPE::isTrans && B_TYPE::isTrans) {
        kMaxAxis = 2;
    }
    if constexpr (!A_TYPE::isTrans && B_TYPE::isTrans) {
        kMaxAxis = mmCFG.basicM > mmCFG.basicN ? 1 : 2;
    }
    return kMaxAxis;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ constexpr int32_t GetIterateOrder(const L1Status &l1Status, const MatmulConfig &mmCFG)
{
    const int32_t reduceSize = GetReduceC0Size<typename A_TYPE::T>();
    bool fullkAL1Load = static_cast<int32_t>(mmCFG.singleCoreK) <= l1Status.kAL1 * reduceSize;
    bool fullkBL1Load = static_cast<int32_t>(mmCFG.singleCoreK) <= l1Status.kBL1 * reduceSize;

    // if KAL1 and KBL1 both can not be full loaded, then select m or n which is no matter
    if (!fullkAL1Load && !fullkBL1Load) {
        return 0;
    } else if (fullkAL1Load && !fullkBL1Load) {
        // if KAL1 is full loaded, then select the order N first
        return 1;
    } else if (!fullkAL1Load && fullkBL1Load) {
        // if KBL1 is full loaded, then select the order M first
        return 0;
    } else {
        // if AL1LoadSize less than BL1LoadSize, then select order N first, vice versa.
        int32_t mLoop = CeilNoLog<int32_t>(mmCFG.singleCoreM, l1Status.mAL1 * mmCFG.basicM);
        int32_t nLoop = CeilNoLog<int32_t>(mmCFG.singleCoreN, l1Status.nBL1 * mmCFG.basicN);
        int32_t aL1LoadSize = mmCFG.singleCoreM + mmCFG.singleCoreN * mLoop;
        int32_t bL1LoadSize = mmCFG.singleCoreN + mmCFG.singleCoreM * nLoop;
        return aL1LoadSize < bL1LoadSize ? 1 : 0;
    }
}

__aicore__ constexpr int32_t GetL0ADb(const MatmulConfig &mmCFG, uint32_t l0ASize)
{
    return (mmCFG.basicM * Impl::C0_BYTE_SIZE > l0ASize / Impl::DB_ON) ? Impl::DB_OFF : Impl::DB_ON;
}

__aicore__ constexpr int32_t GetL0BDb(const MatmulConfig &mmCFG, uint32_t l0BSize)
{
    return (mmCFG.basicN * Impl::C0_BYTE_SIZE > l0BSize / Impl::DB_ON) ? Impl::DB_OFF : Impl::DB_ON;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ constexpr int32_t GetL1UsedSize(const MatmulConfig &mmCFG, const L1Status &l1Status,
    int32_t depthA1, int32_t depthB1)
{
    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    int32_t sharedl1Size = 0;
    if constexpr (!PhyPosIsL1(A_TYPE::pos)) {
        sharedl1Size += depthA1 * mmCFG.basicM * mmCFG.basicK * GetBitSize<SrcAT>() / ONE_BYTE_BIT_SIZE;
    }
    if constexpr (!PhyPosIsL1(B_TYPE::pos)) {
        if constexpr (IsSameTypeV<SrcAT, SrcBT>) {
            sharedl1Size += depthB1 * mmCFG.basicN * mmCFG.basicK *
                GetBitSize<SrcBT>() / ONE_BYTE_BIT_SIZE;
        } else {
            // A16W8 w8 use same with A_TYPE
            sharedl1Size += depthB1 * mmCFG.basicN * mmCFG.basicK *
                GetBitSize<SrcAT>() / ONE_BYTE_BIT_SIZE;
        }
    }
    if (mmCFG.enableSetBias) {
        if constexpr (!PhyPosIsL1(BIAS_TYPE::pos)) {
            sharedl1Size += mmCFG.basicN * GetBitSize<BiasT>() / ONE_BYTE_BIT_SIZE;
        }
    }
    sharedl1Size += GetDeQuantSize(l1Status, mmCFG);
    return sharedl1Size;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ constexpr int32_t GetL1UsedSize(const MatmulConfig &mmCFG, int32_t depthA1, int32_t depthB1)
{
    int32_t sharedl1Size = 0;
    if constexpr (!PhyPosIsL1(A_TYPE::pos)) {
        sharedl1Size += depthA1 * mmCFG.basicM * mmCFG.basicK * GetBitSize<typename A_TYPE::T>() / ONE_BYTE_BIT_SIZE;
    }
    if constexpr (!PhyPosIsL1(B_TYPE::pos)) {
        if constexpr (IsSameTypeV<typename A_TYPE::T, typename B_TYPE::T>) {
            sharedl1Size += depthB1 * mmCFG.basicN * mmCFG.basicK *
                GetBitSize<typename B_TYPE::T>() / ONE_BYTE_BIT_SIZE;
        } else {
            // A16W8 w8 use same with A_TYPE
            sharedl1Size += depthB1 * mmCFG.basicN * mmCFG.basicK *
                GetBitSize<typename A_TYPE::T>() / ONE_BYTE_BIT_SIZE;
        }
    }
    if (mmCFG.enableSetBias) {
        if constexpr (!PhyPosIsL1(BIAS_TYPE::pos)) {
            sharedl1Size += mmCFG.basicN * GetBitSize<typename BIAS_TYPE::T>() / ONE_BYTE_BIT_SIZE;
        }
    }
    if (mmCFG.enableQuantVector) {
        sharedl1Size += depthB1 * mmCFG.basicN * sizeof(uint64_t);
    }
    return sharedl1Size;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ constexpr int32_t GetTransLength(const MatmulConfig &mmCFG, const L1Status &l1Status)
{
    int32_t a1Length = 0;
    int32_t b1Length = 0;
    int32_t c1Length = 0;
    int32_t biasLength = GetBiasL1Size<BIAS_TYPE>(l1Status, mmCFG);
    // A matrix ND2NZ
    if constexpr (A_TYPE::format == CubeFormat::ND && (A_TYPE::pos == TPosition::VECIN ||
        A_TYPE::pos == TPosition::VECCALC || A_TYPE::pos == TPosition::VECOUT)) {
        a1Length = mmCFG.singleCoreM * mmCFG.singleCoreK * GetBitSize<typename A_TYPE::T>() / ONE_BYTE_BIT_SIZE;
    }
    // B matrix ND2NZ
    if constexpr (B_TYPE::format == CubeFormat::ND && (B_TYPE::pos == TPosition::VECIN ||
        B_TYPE::pos == TPosition::VECCALC || B_TYPE::pos == TPosition::VECOUT)) {
        // A16W8, B type in L1 is same with ATYPE, so use A_TYPE::T
        b1Length = mmCFG.singleCoreK * mmCFG.singleCoreN * GetBitSize<typename A_TYPE::T>() / ONE_BYTE_BIT_SIZE;
    }
    // C matrix ND2NZ
    if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::pos == TPosition::GM) {
        c1Length = mmCFG.basicM * mmCFG.basicN * GetBitSize<typename C_TYPE::T>() / ONE_BYTE_BIT_SIZE;
    }
    return MaxValue<int32_t>(a1Length, b1Length, c1Length, biasLength);
}
} // namespace AscendC
#endif // _MATMUL_CONSTANT_TILING_UTILS_H_