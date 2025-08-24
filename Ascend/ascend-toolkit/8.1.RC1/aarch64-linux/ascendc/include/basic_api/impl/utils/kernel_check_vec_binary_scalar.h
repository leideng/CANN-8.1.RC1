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
 * \file kernel_check_vec_binary_scalar.h
 * \brief
 */
#ifndef ASCENDC_MODULE_CHECK_VEC_BINARY_SACLAR_H
#define ASCENDC_MODULE_CHECK_VEC_BINARY_SACLAR_H

#if ASCENDC_CPU_DEBUG
#include "tikcpp_check_util.h"
#include "kernel_common.h"
#include "kernel_struct_binary.h"
#include "kernel_struct_unary.h"

namespace AscendC {
template <typename T, typename U>
bool CheckFuncVecBinaryScalarCmp(const LocalTensor<U>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, const char* intriName)
{
    (void)(scalarValue);
    check::VecBinaryScalarApiParams chkParams { static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.srcBlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.srcRepStride),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(U)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFuncVecBinaryScalarCmpImpl(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFuncVecBinaryScalarCmp(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    T src1Scalar, const int32_t& calCount, const char* intriName)
{
    check::VecBinaryScalarApiParams chkParams {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(U)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint32_t>(calCount) };
    return CheckFuncVecBinaryScalarCmpImpl(chkParams, intriName);
}

template <typename T, typename U>
bool CheckFunVecBinaryScalar(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, const char* intriName)
{
    using PrimType = PrimT<T>;
    (void)(scalarValue);
    check::VecBinaryScalarApiParams chkParams { static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.srcBlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.srcRepStride),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimType)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(PrimType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFunVecBinaryScalarImplForMaskArray(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFunVecBinaryScalar(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams, const char* intriName)
{
    using PrimType = PrimT<T>;
    (void)(scalarValue);
    check::VecBinaryScalarApiParams chkParams { static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.src0BlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.src0RepStride),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimType)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(PrimType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFunVecBinaryScalarImplForMaskArray(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFunVecBinaryScalar(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, const char* intriName)
{
    using PrimType = PrimT<T>;
    (void)(scalarValue);
    check::VecBinaryScalarApiParams chkParams { static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.srcBlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.srcRepStride),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimType)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(PrimType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFunVecBinaryScalarImpl(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFunVecBinaryScalar(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams, const char* intriName)
{
    using PrimType = PrimT<T>;
    (void)(scalarValue);
    check::VecBinaryScalarApiParams chkParams { static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.src0BlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.src0RepStride),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimType)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(PrimType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFunVecBinaryScalarImpl(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFunVecBinaryScalar(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount, const char* intriName)
{
    using PrimType = PrimT<T>;
    (void)(scalarValue);
    check::VecBinaryScalarApiParams chkParams { static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimType)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(PrimType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        static_cast<uint32_t>(calCount) };
    return CheckFunVecBinaryScalarImpl(chkParams, intriName);
}

template <typename T, typename U>
bool CheckFunVecBinaryScalarDiffType(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
    const U& scalarValue, const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams,
    const char* intriName)
{
    (void)(scalarValue);
    check::VecBinaryScalarApiParams chkParams { static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.srcBlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.srcRepStride),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFunVecBinaryScalarImplForMaskArray(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFunVecBinaryScalarDiffType(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
    const U& scalarValue, const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams,
    const char* intriName)
{
    (void)(scalarValue);
    check::VecBinaryScalarApiParams chkParams { static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.srcBlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.srcRepStride),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFunVecBinaryScalarImpl(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFunVecBinaryScalarDiffType(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
    const U& scalarValue, const int32_t& calCount, const char* intriName)
{
    (void)(scalarValue);
    check::VecBinaryScalarApiParams chkParams { static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        static_cast<uint32_t>(calCount) };
    return CheckFunVecBinaryScalarImpl(chkParams, intriName);
}
} // namespace AscendC
#endif

#endif