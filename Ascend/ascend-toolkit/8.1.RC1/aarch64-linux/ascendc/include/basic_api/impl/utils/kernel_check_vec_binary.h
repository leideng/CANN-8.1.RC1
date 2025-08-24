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
 * \file kernel_check_vec_binary.h
 * \brief
 */
#ifndef ASCENDC_MODULE_CHECK_VEC_BINARY_H
#define ASCENDC_MODULE_CHECK_VEC_BINARY_H

#if ASCENDC_CPU_DEBUG
#include "tikcpp_check_util.h"
#include "kernel_common.h"
#include "kernel_struct_binary.h"

namespace AscendC {
template <typename T>
bool CheckFuncVecBinary(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, const uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams, const char* intriName)
{
    using PrimType = PrimT<T>;
    check::VecBinaryApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.src0BlkStride),
        static_cast<uint16_t>(repeatParams.src1BlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.src0RepStride),
        static_cast<uint16_t>(repeatParams.src1RepStride),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimType)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(PrimType)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(PrimType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFuncVecBinaryImplForMaskArray(chkParams, mask, intriName);
}

template <typename T>
bool CheckFuncVecBinary(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, const uint64_t mask, const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams, const char* intriName)
{
    using PrimType = PrimT<T>;
    check::VecBinaryApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.src0BlkStride),
        static_cast<uint16_t>(repeatParams.src1BlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.src0RepStride),
        static_cast<uint16_t>(repeatParams.src1RepStride),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimType)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(PrimType)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(PrimType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFuncVecBinaryImpl(chkParams, mask, intriName);
}

template <typename T>
bool CheckFuncVecBinary(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, const int32_t& calCount, const char* intriName)
{
    using PrimType = PrimT<T>;
    check::VecBinaryApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint32_t>(sizeof(PrimType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimType)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(PrimType)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(PrimType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()),
        static_cast<uint32_t>(calCount) };
    return CheckFuncVecBinaryImpl(chkParams, intriName);
}

template <typename T, typename U>
bool CheckFuncSelectVec(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask, const LocalTensor<T>& src0Local,
     const LocalTensor<T>& src1Local, uint64_t mask[], uint8_t repeatTimes, const BinaryRepeatParams& repeatParams,
     const char* intriName)
{
    using PrimDstSrcType = PrimT<T>;
    using PrimSelMaskType = PrimT<U>;
    check::VecSelectApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(selMask.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.src0BlkStride),
        static_cast<uint16_t>(repeatParams.src1BlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.src0RepStride),
        static_cast<uint16_t>(repeatParams.src1RepStride),
        static_cast<uint32_t>(sizeof(PrimDstSrcType)),
        static_cast<uint32_t>(sizeof(PrimSelMaskType)),
        static_cast<uint32_t>(sizeof(PrimDstSrcType)),
        static_cast<uint32_t>(sizeof(PrimDstSrcType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimDstSrcType)),
        static_cast<uint64_t>(selMask.GetSize() * sizeof(PrimSelMaskType)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(PrimDstSrcType)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(PrimDstSrcType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(selMask.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFuncVecSelectImplForMaskArray(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFuncSelectVec(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask, const LocalTensor<T>& src0Local,
     const LocalTensor<T>& src1Local, uint64_t mask, uint8_t repeatTimes, const BinaryRepeatParams& repeatParams,
     const char* intriName)
{
    using PrimDstSrcType = PrimT<T>;
    using PrimSelMaskType = PrimT<U>;
    check::VecSelectApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(selMask.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.src0BlkStride),
        static_cast<uint16_t>(repeatParams.src1BlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.src0RepStride),
        static_cast<uint16_t>(repeatParams.src1RepStride),
        static_cast<uint32_t>(sizeof(PrimDstSrcType)),
        static_cast<uint32_t>(sizeof(PrimSelMaskType)),
        static_cast<uint32_t>(sizeof(PrimDstSrcType)),
        static_cast<uint32_t>(sizeof(PrimDstSrcType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimDstSrcType)),
        static_cast<uint64_t>(selMask.GetSize() * sizeof(PrimSelMaskType)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(PrimDstSrcType)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(PrimDstSrcType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(selMask.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFuncVecSelectImpl(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFuncSelectVec(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, const int32_t& calCount, const char* intriName)
{
    using PrimDstSrcType = PrimT<T>;
    using PrimSelMaskType = PrimT<U>;
    check::VecSelectApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(selMask.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(PrimDstSrcType)),
        static_cast<uint32_t>(sizeof(PrimSelMaskType)),
        static_cast<uint32_t>(sizeof(PrimDstSrcType)),
        static_cast<uint32_t>(sizeof(PrimDstSrcType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimDstSrcType)),
        static_cast<uint64_t>(selMask.GetSize() * sizeof(PrimSelMaskType)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(PrimDstSrcType)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(PrimDstSrcType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(selMask.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()),
        static_cast<uint32_t>(calCount) };
    return CheckFuncVecSelectImpl(chkParams, intriName);
}

template <typename T, typename U>
bool CheckFuncVecBinaryDiffType(const LocalTensor<T>& dstLocal, const LocalTensor<U>& src0Local,
    const LocalTensor<U>& src1Local, const uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams, const char* intriName)
{
    using PrimDstType = PrimT<T>;
    using PrimSrcType = PrimT<U>;
    check::VecBinaryApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.src0BlkStride),
        static_cast<uint16_t>(repeatParams.src1BlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.src0RepStride),
        static_cast<uint16_t>(repeatParams.src1RepStride),
        static_cast<uint32_t>(sizeof(PrimDstType)),
        static_cast<uint32_t>(sizeof(PrimSrcType)),
        static_cast<uint32_t>(sizeof(PrimSrcType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimDstType)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(PrimSrcType)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(PrimSrcType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFuncVecBinaryImplForMaskArray(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFuncVecBinaryDiffType(const LocalTensor<T>& dstLocal, const LocalTensor<U>& src0Local,
    const LocalTensor<U>& src1Local, const uint64_t mask, const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams, const char* intriName)
{
    using PrimDstType = PrimT<T>;
    using PrimSrcType = PrimT<U>;
    check::VecBinaryApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.src0BlkStride),
        static_cast<uint16_t>(repeatParams.src1BlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.src0RepStride),
        static_cast<uint16_t>(repeatParams.src1RepStride),
        static_cast<uint32_t>(sizeof(PrimDstType)),
        static_cast<uint32_t>(sizeof(PrimSrcType)),
        static_cast<uint32_t>(sizeof(PrimSrcType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimDstType)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(PrimSrcType)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(PrimSrcType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFuncVecBinaryImpl(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFuncVecBinaryDiffType(const LocalTensor<T>& dstLocal, const LocalTensor<U>& src0Local,
    const LocalTensor<U>& src1Local, const int32_t& calCount, const char* intriName)
{
    using PrimDstType = PrimT<T>;
    using PrimSrcType = PrimT<U>;
    check::VecBinaryApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(PrimDstType)),
        static_cast<uint32_t>(sizeof(PrimSrcType)),
        static_cast<uint32_t>(sizeof(PrimSrcType)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimDstType)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(PrimSrcType)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(PrimSrcType)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()),
        static_cast<uint32_t>(calCount) };
    return CheckFuncVecBinaryImpl(chkParams, intriName);
}

template <typename T, typename U>
bool CheckFuncVecBinaryCmp(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, const uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams, const char* intriName)
{
    check::VecBinaryApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.src0BlkStride),
        static_cast<uint16_t>(repeatParams.src1BlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.src0RepStride),
        static_cast<uint16_t>(repeatParams.src1RepStride),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(U)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFuncVecBinaryCmpImplForMaskArray(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFuncVecBinaryCmp(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, const uint64_t mask, const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams, const char* intriName)
{
    check::VecBinaryApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.src0BlkStride),
        static_cast<uint16_t>(repeatParams.src1BlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.src0RepStride),
        static_cast<uint16_t>(repeatParams.src1RepStride),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(U)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFuncVecBinaryCmpImpl(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFuncVecBinaryCmp(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, const int32_t& calCount, const char* intriName)
{
    check::VecBinaryApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(U)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()),
        static_cast<uint32_t>(calCount) };
    return CheckFuncVecBinaryCmpImpl(chkParams, intriName);
}

template <typename T>
bool CheckFuncVecBinaryCmpRgt(const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local, const uint64_t mask[],
    const BinaryRepeatParams& repeatParams, const char* intriName)
{
    check::VecCmpRgtApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        static_cast<uint16_t>(repeatParams.src0BlkStride),
        static_cast<uint16_t>(repeatParams.src1BlkStride),
        static_cast<uint16_t>(repeatParams.src0RepStride),
        static_cast<uint16_t>(repeatParams.src1RepStride),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(T)),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFuncVecCmpRgtImplForMaskArray(chkParams, mask, intriName);
}

template <typename T>
bool CheckFuncVecBinaryCmpRgt(const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local, const uint64_t mask,
    const BinaryRepeatParams& repeatParams, const char* intriName)
{
    check::VecCmpRgtApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        static_cast<uint16_t>(repeatParams.src0BlkStride),
        static_cast<uint16_t>(repeatParams.src1BlkStride),
        static_cast<uint16_t>(repeatParams.src0RepStride),
        static_cast<uint16_t>(repeatParams.src1RepStride),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(T)),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFuncVecCmpRgtImpl(chkParams, mask, intriName);
}

} // namespace AscendC
#endif

#endif