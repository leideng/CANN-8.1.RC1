/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file kernel_tensor_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_TENSOR_IMPL_H
#define ASCENDC_MODULE_TENSOR_IMPL_H
#include "kernel_tensor.h"

namespace AscendC {
// CPU Impl
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
inline uint8_t* GetBaseAddrCpu(int8_t logicPos);
#endif
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
template <typename T> LocalTensor<T>::LocalTensor(TBuffAddr& address)
{
    this->address_ = address;
}

template <typename T> LocalTensor<T>::LocalTensor(const LocalTensor<T>& other)
{
    os_.str("");
    this->address_ = other.address_;
#ifndef __DAV_C220_CUBE__
    if constexpr (IsSameType<PrimType, T>::value) {
        this->shapeInfo_ = other.shapeInfo_;
    }
#endif
}

template <typename T> LocalTensor<T> LocalTensor<T>::operator = (const LocalTensor<T>& other)
{
    if (this != &other) {
        os_.str("");
        this->address_ = other.address_;
#ifndef __DAV_C220_CUBE__
        if constexpr (IsSameType<PrimType, T>::value) {
            this->shapeInfo_ = other.shapeInfo_;
        }
#endif
    }
    return *this;
}

template <typename T>
typename LocalTensor<T>::PrimType* LocalTensor<T>::GetPhyAddr(const uint32_t offset) const
{
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        ASCENDC_ASSERT((this->address_.dataLen * INT4_TWO > (offset / INT4_TWO)), {
        KERNEL_LOG(KERNEL_ERROR, "offset is %u, which can not be larger than data len %u", offset,
            static_cast<uint32_t>(this->address_.dataLen * INT4_TWO));
        });
        ASCENDC_ASSERT((offset % INT4_TWO == 0), {
            KERNEL_LOG(KERNEL_ERROR, "The offset for int4b_t GetPhyAddr should be an even num.");});
        return reinterpret_cast<int4b_t *>(this->address_.absAddr) + offset / INT4_TWO;
    } else {
        ASCENDC_ASSERT((this->address_.dataLen > (offset * sizeof(PrimType))), {
        KERNEL_LOG(KERNEL_ERROR, "offset is %u, which can not be larger than data len %u", offset,
            static_cast<uint32_t>(this->address_.dataLen / sizeof(PrimType)));
        });
        return reinterpret_cast<PrimType*>(this->address_.absAddr) + offset;
    }
}
template <typename T> typename LocalTensor<T>::PrimType* LocalTensor<T>::GetPhyAddr() const
{
    return GetPhyAddr(0);
}

template <typename T>
__inout_pipe__(S) typename LocalTensor<T>::PrimType LocalTensor<T>::GetValue(const uint32_t offset) const
{
    if ASCEND_IS_AIC {
        if (GetPhyType(AscendC::TPosition(this->GetPosition())) == Hardware::UB) {
            return 0;
        }
    }
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        ASCENDC_ASSERT((this->address_.dataLen * INT4_TWO > (offset / INT4_TWO)), {
        KERNEL_LOG(KERNEL_ERROR, "offset is %u, which can not be larger than data len %u", offset,
            static_cast<uint32_t>(this->address_.dataLen * INT4_TWO));
        });

        LocalTensor<uint8_t> tmpLocalTensor = this->ReinterpretCast<uint8_t>();
        uint8_t val = tmpLocalTensor.GetValue(offset / INT4_TWO);
        return static_cast<int4b_t>(val >> (4 * (offset % INT4_TWO)));
    } else {
        ASCENDC_ASSERT((this->address_.dataLen > (offset * sizeof(PrimType))), {
        KERNEL_LOG(KERNEL_ERROR, "offset is %u, which can not be larger than data len %u", offset,
            static_cast<uint32_t>(this->address_.dataLen / sizeof(PrimType)));
        });
        return *(GetPhyAddr(offset));
    }
}

template <typename T>
__inout_pipe__(S) typename LocalTensor<T>::PrimType& LocalTensor<T>::operator()(const uint32_t offset) const
{
    ASCENDC_ASSERT((this->address_.dataLen > (offset * sizeof(PrimType))), {
        KERNEL_LOG(KERNEL_ERROR, "offset is %u, which can not be larger than data len %u", offset,
            static_cast<uint32_t>(this->address_.dataLen / sizeof(PrimType)));
    });
    return *(GetPhyAddr(offset));
}
template <typename T>
template <typename CAST_T> __aicore__ inline LocalTensor<CAST_T> LocalTensor<T>::ReinterpretCast() const
{
    LocalTensor<CAST_T> tensorOut;
    tensorOut.address_.logicPos = static_cast<uint8_t>(this->GetPosition());
    tensorOut.address_.bufferHandle = this->GetBufferHandle();
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        tensorOut.address_.dataLen = this->GetSize() / INT4_TWO;
    } else {
        tensorOut.address_.dataLen = this->GetSize() * sizeof(PrimType);
    }
    tensorOut.address_.bufferAddr = this->address_.bufferAddr;
    tensorOut.address_.absAddr = this->address_.absAddr;
    return tensorOut;
}

template <typename T>
template <typename T1> __inout_pipe__(S) void LocalTensor<T>::SetValue(const uint32_t index, const T1 value) const
{
    if ASCEND_IS_AIC {
        if (GetPhyType(AscendC::TPosition(this->GetPosition())) == Hardware::UB) {
            return;
        }
    }
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        ASCENDC_ASSERT((this->address_.dataLen * INT4_TWO > (index / INT4_TWO)), {
        KERNEL_LOG(KERNEL_ERROR, "index is %u, which can not be larger than data len %u", index,
            static_cast<uint32_t>(this->address_.dataLen * INT4_TWO));
        });

        LocalTensor<uint8_t> tmpLocalTensor = this->ReinterpretCast<uint8_t>();
        uint8_t shift = (index % INT4_TWO == 0)? 0 : 4;
        uint32_t idx = index / INT4_TWO;
        uint8_t val = tmpLocalTensor.GetValue(idx) & (0xf << (INT4_BIT_NUM - shift));
        tmpLocalTensor.SetValue(idx, val + (value.storage << shift));
    } else {
        ASCENDC_ASSERT((this->address_.dataLen > (index * sizeof(PrimType))), {
        KERNEL_LOG(KERNEL_ERROR, "index is %u, which can not be larger than data len %u", index,
            static_cast<uint32_t>(this->address_.dataLen / sizeof(PrimType)));
        });
        *(GetPhyAddr(index)) = value;
    }
}

template <typename T> LocalTensor<T> LocalTensor<T>::operator[](const uint32_t offset) const
{
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        ASCENDC_ASSERT((this->address_.dataLen > (offset / INT4_TWO)), {
        KERNEL_LOG(KERNEL_ERROR, "offset is %u, which can not be larger than data len %u", offset,
            static_cast<uint32_t>(this->address_.dataLen * INT4_TWO));
        });
    } else {
        ASCENDC_ASSERT((this->address_.dataLen > (offset * sizeof(PrimType))), {
        KERNEL_LOG(KERNEL_ERROR, "offset is %u, which can not be larger than data len %u", offset,
            static_cast<uint32_t>(this->address_.dataLen / sizeof(PrimType)));
        });
    }

    LocalTensor retLocalTensor = *this;
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        retLocalTensor.address_.dataLen -= (offset / INT4_TWO);
        retLocalTensor.address_.absAddr = retLocalTensor.address_.absAddr + offset / INT4_TWO;
        retLocalTensor.address_.bufferAddr = retLocalTensor.address_.bufferAddr + offset / INT4_TWO;
    } else {
        retLocalTensor.address_.dataLen -= (offset * sizeof(PrimType));
        retLocalTensor.address_.absAddr = retLocalTensor.address_.absAddr + offset * sizeof(PrimType);
        retLocalTensor.address_.bufferAddr = retLocalTensor.address_.bufferAddr + offset * sizeof(PrimType);
    }
    retLocalTensor.os_.str("");
    return retLocalTensor;
}

template <typename T>
template <typename T1>
[[deprecated("NOTICE: SetAddrWithOffset has been deprecated and will be removed in the next version. "
    "Please do not use it!")]]
void LocalTensor<T>::SetAddrWithOffset(LocalTensor<T1> &src, uint32_t offset)
{
    this->address_ = src.address_;
    this->address_.bufferAddr += offset * sizeof(PrimT<T1>);
    this->address_.absAddr += offset * sizeof(PrimT<T1>);
}

template <typename T>
[[deprecated("NOTICE: Print has been deprecated and will be removed in the next version. Please do not use "
    "it!")]]
inline void LocalTensor<T>::Print(uint32_t len)
{
    if constexpr (IsSameType<PrimType, half>::value) {
        PrintTypicalFloat(len, sizeof(half));
        return;
    }
#if __CCE_AICORE__ >= 220
    if constexpr (IsSameType<PrimType, bfloat16_t>::value) {
        PrintTypicalFloat(len, sizeof(bfloat16_t));
        return;
    }
    if constexpr (!IsSameType<PrimType, half>::value && !IsSameType<PrimType, bfloat16_t>::value) {
#else
    if constexpr (!IsSameType<PrimType, half>::value) {
#endif
        os_.str("");
        uint32_t printLen = std::min(len, GetSize());
        uint32_t blockNum = ONE_BLK_SIZE / sizeof(PrimType);
        uint32_t rowNum = printLen / blockNum;
        uint32_t residualNum = printLen % blockNum;
        const int32_t width = 4;
        for (uint32_t i = 0; i < rowNum; i++) {
            os_ << std::setw(width) << std::setfill('0') << i * blockNum << " : ";
            for (uint32_t j = 0; j < blockNum; j++) {
                if ((sizeof(PrimType) == sizeof(int8_t)) || (sizeof(PrimType) == sizeof(bool))) {
                    os_ << static_cast<int32_t>(GetValue(i * blockNum + j)) << " ";
                } else {
                    os_ << GetValue(i * blockNum + j) << " ";
                }
            }
            os_ << std::endl;
        }
        if (residualNum != 0) {
            os_ << std::setw(width) << std::setfill('0') << rowNum * blockNum << " : ";
            for (uint32_t i = 0; i < residualNum; i++) {
                if ((sizeof(PrimType) == sizeof(int8_t)) || (sizeof(PrimType) == sizeof(bool))) {
                    os_ << static_cast<int32_t>(GetValue(rowNum * blockNum + i)) << " ";
                } else {
                    os_ << GetValue(rowNum * blockNum + i) << " ";
                }
            }
            os_ << std::endl;
        }
        std::cout << os_.str();
    }
}

template <typename T> inline void LocalTensor<T>::PrintTypicalFloat(uint32_t len, uint32_t dataSize)
{
    os_.str("");
    uint32_t printLen = std::min(len, GetSize());
    uint32_t blockNum = ONE_BLK_SIZE / dataSize;
    uint32_t rowNum = printLen / blockNum;
    uint32_t residualNum = printLen % blockNum;
    const int32_t width = 4;
    for (uint32_t i = 0; i < rowNum; i++) {
        os_ << std::setw(width) << std::setfill('0') << i * blockNum << " : ";
        for (uint32_t j = 0; j < blockNum; j++) {
            os_ << GetValue(i * blockNum + j).ToFloat() << " ";
        }
        os_ << std::endl;
    }
    if (residualNum != 0) {
        os_ << std::setw(width) << std::setfill('0') << rowNum * blockNum << " : ";
        for (uint32_t i = 0; i < residualNum; i++) {
            os_ << GetValue(rowNum * blockNum + i).ToFloat() << " ";
        }
        os_ << std::endl;
    }
    std::cout << os_.str();
}

template <>
[[deprecated("NOTICE: Print has been deprecated and will be removed in the next version. Please do not use "
    "it!")]]
inline void LocalTensor<half>::Print(uint32_t len)
{
    PrintTypicalFloat(len, sizeof(half));
}

#if __CCE_AICORE__ >= 220
template <>
[[deprecated("NOTICE: Print has been deprecated and will be removed in the next version. Please do not use "
"it!")]]
inline void LocalTensor<bfloat16_t>::Print(uint32_t len)
{
PrintTypicalFloat(len, sizeof(bfloat16_t));
}
#endif

template <typename T>
[[deprecated("NOTICE: Print has been deprecated and will be removed in the next version. Please do not use "
"it!")]]
inline void LocalTensor<T>::Print()
{
Print(GetSize());
}

template <typename T> LocalTensor<T>::~LocalTensor() {}

template <typename T>
[[deprecated("NOTICE: ToFile has been deprecated and will be removed in the next version. Please do not use "
"it!")]]
int32_t LocalTensor<T>::ToFile(const std::string &fileName) const
{
return TensorWriteFile(fileName, reinterpret_cast<const PrimType *>(GetPhyAddr()), GetSize() * sizeof(PrimType));
}
// Npu Impl
#else
template <typename T> __aicore__ inline uint64_t LocalTensor<T>::GetPhyAddr() const
{
    return GetPhyAddr(0);
}
template <typename T> __aicore__ inline uint64_t LocalTensor<T>::GetPhyAddr(const uint32_t offset) const
{
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        return this->address_.bufferAddr + offset / INT4_TWO;
    } else {
        return this->address_.bufferAddr + offset * sizeof(PrimType);
    }
}
template <typename T> __aicore__ inline __inout_pipe__(S)
    typename LocalTensor<T>::PrimType LocalTensor<T>::GetValue(const uint32_t index) const
{
    if ASCEND_IS_AIC {
        if (GetPhyType(AscendC::TPosition(this->GetPosition())) == Hardware::UB) {
            return 0;
        }
    }
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        LocalTensor<uint8_t> tmpLocalTensor = this->ReinterpretCast<uint8_t>();
        uint8_t val = tmpLocalTensor.GetValue(index / INT4_TWO);
        return static_cast<int4b_t>(val >> (INT4_BIT_NUM * (index % INT4_TWO)));
    } else {
        return *(reinterpret_cast<__ubuf__ PrimType*>(GetPhyAddr(index)));
    }
}
template <typename T> __aicore__ inline __inout_pipe__(S)
    __ubuf__ typename LocalTensor<T>::PrimType& LocalTensor<T>::operator()(const uint32_t offset) const
{
    return *(reinterpret_cast<__ubuf__ PrimType*>(GetPhyAddr(offset)));
}

template <typename T>
template <typename CAST_T> __aicore__ inline __sync_alias__ LocalTensor<CAST_T> LocalTensor<T>::ReinterpretCast() const
{
    LocalTensor<CAST_T> tensorOut;
    tensorOut.address_.logicPos = static_cast<uint8_t>(this->GetPosition());
    tensorOut.address_.bufferHandle = this->GetBufferHandle();
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        tensorOut.address_.dataLen = this->GetSize() / INT4_TWO;
    } else {
        tensorOut.address_.dataLen = this->GetSize() * sizeof(PrimType);
    }
    tensorOut.address_.bufferAddr = this->address_.bufferAddr;
    return tensorOut;
}

template <typename T>
template <typename T1> __aicore__ inline __inout_pipe__(S)
    void LocalTensor<T>::SetValue(const uint32_t index, const T1 value) const
{
    if ASCEND_IS_AIC {
        if (GetPhyType(AscendC::TPosition(this->GetPosition())) == Hardware::UB) {
            return;
        }
    }
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        LocalTensor<uint8_t> tmpLocalTensor = this->ReinterpretCast<uint8_t>();
        uint8_t mask = (index % INT4_TWO == 0)? 0xf0 : 0xf;
        uint32_t idx = index / INT4_TWO;
        uint8_t val = tmpLocalTensor.GetValue(idx) & mask;
        uint8_t shift = (index % INT4_TWO == 0)? 0 : INT4_BIT_NUM;
        tmpLocalTensor.SetValue(idx, val + (value.storage << shift));
    } else {
        *(reinterpret_cast<__ubuf__ PrimType*>(static_cast<uint64_t>(this->address_.bufferAddr))
            + index) = static_cast<PrimType>(value);
    }
}

template <typename T> __aicore__ inline LocalTensor<T> LocalTensor<T>::operator[](const uint32_t offset) const
{
    LocalTensor retLocalTensor = *this;
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        retLocalTensor.address_.dataLen -= (offset / INT4_TWO);
        retLocalTensor.address_.bufferAddr = retLocalTensor.address_.bufferAddr + offset / INT4_TWO;
    } else {
        retLocalTensor.address_.dataLen -= (offset * sizeof(PrimType));
        retLocalTensor.address_.bufferAddr = retLocalTensor.address_.bufferAddr + offset * sizeof(PrimType);
    }
    return retLocalTensor;
}

template <typename T>
template <typename T1>
[[deprecated("NOTICE: SetAddrWithOffset has been deprecated and will be removed in the next version. "
    "Please do not use it!")]]
__aicore__ inline void LocalTensor<T>::SetAddrWithOffset(LocalTensor<T1> &src, uint32_t offset)
{
    this->address_ = src.address_;
    this->address_.bufferAddr += offset * sizeof(PrimT<T1>);
}
#endif
template <typename T> __aicore__ inline int32_t LocalTensor<T>::GetPosition() const
{
    return this->address_.logicPos;
}

template <typename T> __aicore__ inline void LocalTensor<T>::SetSize(const uint32_t size)
{
#if ASCENDC_CPU_DEBUG
    uint32_t len = IsSameType<PrimType, int4b_t>::value ? size / INT4_TWO : size * sizeof(PrimType);
    ASCENDC_ASSERT(((this->address_.absAddr -
        (uint8_t*)(GetBaseAddrCpu(int8_t(AscendC::TPosition(this->address_.logicPos)))) + len) <=
        ConstDefiner::Instance().bufferInitLen.at(ConstDefiner::Instance().positionHardMap.at(
        AscendC::TPosition(this->address_.logicPos)))), {KERNEL_LOG(KERNEL_ERROR,
                "Failed to check param size value in SetSize, current value is %d, buffer overflow", len);});
#endif
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        this->address_.dataLen = size / INT4_TWO;
    } else {
        this->address_.dataLen = size * sizeof(PrimType);
    }
}
template <typename T>
__aicore__ inline uint32_t LocalTensor<T>::GetSize() const
{
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        return this->address_.dataLen * INT4_TWO;
    } else {
        return this->address_.dataLen / sizeof(PrimType);
    }
}

template <typename T>
[[deprecated("NOTICE: GetLength has been deprecated and will be removed in the next version. Please do not use "
                "it!")]]
__aicore__ inline uint32_t LocalTensor<T>::GetLength() const
{
    return this->address_.dataLen;
}

template <typename T>
[[deprecated("NOTICE: SetBufferLen has been deprecated and will be removed in the next version. Please do not use "
                "it!")]]
__aicore__ inline void LocalTensor<T>::SetBufferLen(uint32_t dataLen)
{
    this->address_.dataLen = dataLen;
}
template <typename T> __aicore__ inline void LocalTensor<T>::SetUserTag(const TTagType tag)
{
    auto ptr = reinterpret_cast<TBufType*>(this->address_.bufferHandle);
    ASCENDC_ASSERT((ptr != nullptr),
                    { KERNEL_LOG(KERNEL_ERROR, "ptr can not be nullptr"); });
    ptr->usertag = tag;
}
template <typename T> __aicore__ inline TTagType LocalTensor<T>::GetUserTag() const
{
    auto ptr = reinterpret_cast<TBufType*>(this->address_.bufferHandle);
    ASCENDC_ASSERT((ptr != nullptr),
                    { KERNEL_LOG(KERNEL_ERROR, "ptr can not be nullptr"); });
    return ptr->usertag;
}
// symbol override
template <typename T> __aicore__ inline void LocalTensor<T>::operator = (const SymbolOverrideAdd<T>& symbolOverride)
{
    symbolOverride.Process(*this);
}
template <typename T> __aicore__ inline void LocalTensor<T>::operator = (const SymbolOverrideSub<T>& symbolOverride)
{
    symbolOverride.Process(*this);
}
template <typename T> __aicore__ inline void LocalTensor<T>::operator = (const SymbolOverrideMul<T>& symbolOverride)
{
    symbolOverride.Process(*this);
}
template <typename T> __aicore__ inline void LocalTensor<T>::operator = (const SymbolOverrideDiv<T>& symbolOverride)
{
    symbolOverride.Process(*this);
}
template <typename T> __aicore__ inline void LocalTensor<T>::operator = (const SymbolOverrideOr<T>& symbolOverride)
{
    symbolOverride.Process(*this);
}
template <typename T> __aicore__ inline void LocalTensor<T>::operator = (const SymbolOverrideAnd<T>& symbolOverride)
{
    symbolOverride.Process(*this);
}
template <typename T> __aicore__ inline void
    LocalTensor<T>::operator = (const SymbolOverrideCompare<float>& symbolOverride)
{
    symbolOverride.Process(*this);
}
template <typename T> __aicore__ inline void
    LocalTensor<T>::operator = (const SymbolOverrideCompare<half>& symbolOverride)
{
    symbolOverride.Process(*this);
}

template <typename T> __aicore__ inline SymbolOverrideAdd<T>
    LocalTensor<T>::operator + (const LocalTensor<T>& src1Tensor) const
{
    return SymbolOverrideAdd<T>(*this, src1Tensor);
}
template <typename T> __aicore__ inline SymbolOverrideSub<T>
    LocalTensor<T>::operator - (const LocalTensor<T>& src1Tensor) const
{
    return SymbolOverrideSub<T>(*this, src1Tensor);
}
template <typename T> __aicore__ inline SymbolOverrideMul<T>
    LocalTensor<T>::operator *(const LocalTensor<T>& src1Tensor) const
{
    return SymbolOverrideMul<T>(*this, src1Tensor);
}
template <typename T> __aicore__ inline SymbolOverrideDiv<T>
    LocalTensor<T>::operator / (const LocalTensor<T>& src1Tensor) const
{
    return SymbolOverrideDiv<T>(*this, src1Tensor);
}
template <typename T> __aicore__ inline SymbolOverrideOr<T>
    LocalTensor<T>::operator | (const LocalTensor<T>& src1Tensor) const
{
    return SymbolOverrideOr<T>(*this, src1Tensor);
}
template <typename T> __aicore__ inline SymbolOverrideAnd<T>
    LocalTensor<T>::operator & (const LocalTensor<T>& src1Tensor) const
{
    return SymbolOverrideAnd<T>(*this, src1Tensor);
}
template <typename T> __aicore__ inline SymbolOverrideCompare<T>
    LocalTensor<T>::operator < (const LocalTensor<T>& src1Tensor) const
{
    return SymbolOverrideCompare<T>(*this, src1Tensor, CMPMODE::LT);
}
template <typename T> __aicore__ inline SymbolOverrideCompare<T>
    LocalTensor<T>::operator > (const LocalTensor<T>& src1Tensor) const
{
    return SymbolOverrideCompare<T>(*this, src1Tensor, CMPMODE::GT);
}
template <typename T> __aicore__ inline SymbolOverrideCompare<T>
    LocalTensor<T>::operator != (const LocalTensor<T>& src1Tensor) const
{
    return SymbolOverrideCompare<T>(*this, src1Tensor, CMPMODE::NE);
}
template <typename T> __aicore__ inline SymbolOverrideCompare<T>
    LocalTensor<T>::operator == (const LocalTensor<T>& src1Tensor) const
{
    return SymbolOverrideCompare<T>(*this, src1Tensor, CMPMODE::EQ);
}
template <typename T> __aicore__ inline SymbolOverrideCompare<T>
    LocalTensor<T>::operator <= (const LocalTensor<T>& src1Tensor) const
{
    return SymbolOverrideCompare<T>(*this, src1Tensor, CMPMODE::LE);
}
template <typename T> __aicore__ inline SymbolOverrideCompare<T>
    LocalTensor<T>::operator >= (const LocalTensor<T>& src1Tensor) const
{
    return SymbolOverrideCompare<T>(*this, src1Tensor, CMPMODE::GE);
}
template <typename T> __aicore__ inline void
    LocalTensor<T>::SetShapeInfo(const ShapeInfo& shapeInfo)
{
    static_assert(IsSameType<T, PrimType>::value, "Only primitive type Tensor has shape info!");
#ifndef __DAV_C220_CUBE__
        shapeInfo_ = shapeInfo;
#endif
}
template <typename T> __aicore__ inline ShapeInfo LocalTensor<T>::GetShapeInfo() const
{
    static_assert(IsSameType<T, PrimType>::value, "Only primitive type Tensor has shape info!");
#ifndef __DAV_C220_CUBE__
        return shapeInfo_;
#else
        ShapeInfo tmp;
        return tmp;
#endif
}

template <typename T> __aicore__ inline void
    GlobalTensor<T>::SetGlobalBuffer(__gm__ typename GlobalTensor<T>::PrimType* buffer, uint64_t bufferSize)
{
    if (this->cacheMode_ == CacheMode::CACHE_MODE_NORMAL) {
        this->address_ = buffer;
    } else {
        this->address_ = L2CacheAlter<PrimType, CacheRwMode::RW>(buffer, cacheMode_);
    }
    this->oriAddress_ = buffer;
    bufferSize_ = bufferSize;
}

template <typename T>
__aicore__ inline void GlobalTensor<T>::SetGlobalBuffer(__gm__ typename GlobalTensor<T>::PrimType* buffer)
{
    if (this->cacheMode_ == CacheMode::CACHE_MODE_NORMAL) {
        this->address_ = buffer;
    } else {
        this->address_ = L2CacheAlter<PrimType, CacheRwMode::RW>(buffer, cacheMode_);
    }
    this->oriAddress_ = buffer;
}

template <typename T> __aicore__ inline
    const __gm__ typename GlobalTensor<T>::PrimType* GlobalTensor<T>::GetPhyAddr() const
{
    return this->address_;
}
template <typename T> __aicore__ inline
    __gm__ typename GlobalTensor<T>::PrimType* GlobalTensor<T>::GetPhyAddr(const uint64_t offset) const
{
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        ASCENDC_ASSERT((offset % 2 == 0), {
        KERNEL_LOG(KERNEL_ERROR, "The offset for int4b_t GetPhyAddr should be an even num.");});
        return this->address_ + offset / INT4_TWO;
    } else {
        return this->address_ + offset;
    }
}
template <typename T> __aicore__ inline __inout_pipe__(S)
    typename GlobalTensor<T>::PrimType GlobalTensor<T>::GetValue(const uint64_t offset) const
{
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        __gm__ uint8_t *addr = reinterpret_cast<__gm__ uint8_t *>(this->oriAddress_) + offset / INT4_TWO;
        return static_cast<PrimType>((*addr) >> (INT4_BIT_NUM * (offset % INT4_TWO)));
    } else {
        return this->oriAddress_[offset];
    }
}
template <typename T> __aicore__ inline __inout_pipe__(S)
    __gm__ typename GlobalTensor<T>::PrimType& GlobalTensor<T>::operator()(const uint64_t offset) const
{
    return this->oriAddress_[offset];
}
template <typename T> __aicore__ inline
    void GlobalTensor<T>::SetValue(const uint64_t offset, typename GlobalTensor<T>::PrimType value)
{
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        __gm__ uint8_t *addr = reinterpret_cast<__gm__ uint8_t *>(this->oriAddress_) + offset / INT4_TWO;
        uint8_t mask = (offset % INT4_TWO == 0)? 0xf0 : 0xf;

        uint8_t val = (*addr) & mask;
        uint8_t shift = (offset % INT4_TWO == 0)? 0 : INT4_BIT_NUM;
        *addr = val + (value.storage << shift);
    } else {
        this->oriAddress_[offset] = value;
    }
}
template <typename T> __aicore__ inline uint64_t GlobalTensor<T>::GetSize() const
{
    return bufferSize_;
}
template <typename T> __aicore__ inline GlobalTensor<T> GlobalTensor<T>::operator[](const uint64_t offset) const
{
    GlobalTensor retLocalTensor = *this;
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        retLocalTensor.address_ = retLocalTensor.address_ + offset / INT4_TWO;
        retLocalTensor.oriAddress_ = retLocalTensor.oriAddress_ + offset / INT4_TWO;
    } else {
        retLocalTensor.address_ = retLocalTensor.address_ + offset;
        retLocalTensor.oriAddress_ = retLocalTensor.oriAddress_ + offset;
    }
    return retLocalTensor;
}
template <typename T> __aicore__ inline void GlobalTensor<T>::SetShapeInfo(const ShapeInfo& shapeInfo)
{
    static_assert(IsSameType<T, PrimType>::value, "Only primitive type Tensor has shape info!");
#ifndef __DAV_C220_CUBE__
    shapeInfo_ = shapeInfo;
#endif
}
template <typename T> __aicore__ inline ShapeInfo GlobalTensor<T>::GetShapeInfo() const
{
    static_assert(IsSameType<T, PrimType>::value, "Only primitive type Tensor has shape info!");
#ifndef __DAV_C220_CUBE__
    return shapeInfo_;
#else
    ShapeInfo tmp;
    return tmp;
#endif
}
template <typename T>
template<CacheRwMode rwMode>
__aicore__ inline void GlobalTensor<T>::SetL2CacheHint(CacheMode mode) {
    this->cacheMode_ = mode;
    if (mode == CacheMode::CACHE_MODE_NORMAL) {
        this->address_ = this->oriAddress_;
    } else {
        this->address_ = L2CacheAlter<PrimType, rwMode>(this->oriAddress_, mode);
    }
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    AscendC::OOMAddAddrForL2Cache<PrimType>(this->address_, this->oriAddress_);
#endif // defined(ASCENDC_OOM) && ASCENDC_OOM == 1
}
}
#endif // KERNEL_TENSOR_H

