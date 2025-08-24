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
 * \file kernel_tensor.h
 * \brief
 */
#ifndef KERNEL_TENSOR_H
#define KERNEL_TENSOR_H

#include "kernel_utils.h"
#include "kernel_common.h"
#include "impl/kernel_operator_symbol_override_impl.h"
#include "kernel_tensor_base.h"

namespace AscendC {
/* \brief the shape info of tensor;
 * \note this struct contains the shape info of tensor;
 * info:
 * shape: the tensor's shape
 * shapeDim: the tensor's shape dim
 * originalShape: the tensor's originalShape, for example, this tensor's NZ shape is
 * {32, 32, 16, 16}, but the original shape may be {32*16, 31*16}
 * dataFormat: tensor's format, ND or NZ;
 */
struct ShapeInfo {
public:
    __aicore__ inline ShapeInfo() {}
    __aicore__ inline ShapeInfo(const uint8_t inputShapeDim, const uint32_t inputShape[],
        const uint8_t inputOriginalShapeDim, const uint32_t inputOriginalShape[], const DataFormat inputFormat)
        : shapeDim(inputShapeDim), originalShapeDim(inputOriginalShapeDim), dataFormat(inputFormat)
    {
        ASCENDC_ASSERT((inputShapeDim <= K_MAX_SHAPE_DIM && inputOriginalShapeDim <= K_MAX_SHAPE_DIM), {
            KERNEL_LOG(KERNEL_ERROR,
                "inputShapeDim is %d, inputOriginalShapeDim is %d, which should be less than %d both", inputShapeDim,
                inputOriginalShapeDim, K_MAX_SHAPE_DIM);
        });
        for (int index = 0; index < shapeDim; ++index) {
            shape[index] = inputShape[index];
        }
        for (int index = 0; index < originalShapeDim; ++index) {
            originalShape[index] = inputOriginalShape[index];
        }
    }
    __aicore__ inline ShapeInfo(const uint8_t inputShapeDim, const uint32_t inputShape[], const DataFormat inputFormat)
        : shapeDim(inputShapeDim), originalShapeDim(inputShapeDim), dataFormat(inputFormat)
    {
        ASCENDC_ASSERT((inputShapeDim <= K_MAX_SHAPE_DIM), {
            KERNEL_LOG(KERNEL_ERROR, "inputShapeDim is %u, which should be less than %d",
                                                             inputShapeDim, K_MAX_SHAPE_DIM);
        });
        for (int index = 0; index < shapeDim; ++index) {
            shape[index] = inputShape[index];
            originalShape[index] = inputShape[index];
        }
    }

    __aicore__ inline ShapeInfo(const uint8_t inputShapeDim, const uint32_t inputShape[])
        : shapeDim(inputShapeDim), originalShapeDim(inputShapeDim), dataFormat(DataFormat::ND)
    {
        ASCENDC_ASSERT((inputShapeDim <= K_MAX_SHAPE_DIM), {
            KERNEL_LOG(KERNEL_ERROR, "inputShapeDim is %d, which should be less than %d",
                                                             inputShapeDim, K_MAX_SHAPE_DIM);
        });
        for (int index = 0; index < shapeDim; ++index) {
            shape[index] = inputShape[index];
            originalShape[index] = inputShape[index];
        }
    }
    uint8_t shapeDim;
    uint8_t originalShapeDim;
    uint32_t shape[K_MAX_SHAPE_DIM];
    uint32_t originalShape[K_MAX_SHAPE_DIM];
    DataFormat dataFormat;
};

/* \brief the ShapeInfoParams of shapeInfo_;
 * \note this struct contains typename of shapeInfo_;
 * info:
 * Params: the shapeInfo_'s typename
 */
template <typename U, typename T>
struct ShapeInfoParams {
    __aicore__ ShapeInfoParams() {};
    using Params = ShapeInfo;
};
template <typename T>
struct ShapeInfoParams<TensorTrait<T>, T> {
    __aicore__ ShapeInfoParams() {};
    using Params = int8_t;
};

__aicore__ inline uint64_t GetShapeSize(const ShapeInfo& shapeInfo)
{
    int shapeSize = 1;
    for (int index = 0; index < shapeInfo.shapeDim; ++index) {
        shapeSize *= shapeInfo.shape[index];
    }
    return shapeSize;
}

template <typename T> class LocalTensor : public BaseLocalTensor<T> {
public:
    using PrimType = PrimT<T>;
    __aicore__ inline LocalTensor<T>() {};
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ~LocalTensor();
    explicit LocalTensor<T>(TBuffAddr& address);
    LocalTensor<T>(const LocalTensor<T>& other);
    LocalTensor<T> operator = (const LocalTensor<T>& other);

    PrimType* GetPhyAddr(const uint32_t offset) const;
    PrimType* GetPhyAddr() const;
    __inout_pipe__(S) PrimType GetValue(const uint32_t offset) const;
    __inout_pipe__(S) PrimType& operator()(const uint32_t offset) const;

    template <typename CAST_T> __aicore__ inline LocalTensor<CAST_T> ReinterpretCast() const;
    template <typename T1> __inout_pipe__(S) void SetValue(const uint32_t index, const T1 value) const;
    LocalTensor operator[](const uint32_t offset) const;

    template <typename T1> void SetAddrWithOffset(LocalTensor<T1> &src, uint32_t offset);
    inline void Print();
    inline void Print(uint32_t len);
    int32_t ToFile(const std::string& fileName) const;
#else
    __aicore__ inline uint64_t GetPhyAddr() const;
    __aicore__ inline uint64_t GetPhyAddr(const uint32_t offset) const;
    __aicore__ inline __inout_pipe__(S) PrimType GetValue(const uint32_t index) const;
    __aicore__ inline __inout_pipe__(S) __ubuf__ PrimType& operator()(const uint32_t offset) const;
    template <typename CAST_T> __aicore__ inline LocalTensor<CAST_T> ReinterpretCast() const;
    template <typename T1> __aicore__ inline __inout_pipe__(S)
        void SetValue(const uint32_t index, const T1 value) const;
    __aicore__ inline LocalTensor operator[](const uint32_t offset) const;

    template <typename T1>
    [[deprecated("NOTICE: SetAddrWithOffset has been deprecated and will be removed in the next version. "
        "Please do not use it!")]]
    __aicore__ inline void SetAddrWithOffset(LocalTensor<T1> &src, uint32_t offset);
#endif
    __aicore__ inline int32_t GetPosition() const;
    __aicore__ inline void SetSize(const uint32_t size);
    __aicore__ inline uint32_t GetSize() const;

    [[deprecated("NOTICE: GetLength has been deprecated and will be removed in the next version. Please do not use "
                 "it!")]]
    __aicore__ inline uint32_t GetLength() const;

    [[deprecated("NOTICE: SetBufferLen has been deprecated and will be removed in the next version. Please do not use "
                 "it!")]]
    __aicore__ inline void SetBufferLen(uint32_t dataLen);
    __aicore__ inline void SetUserTag(const TTagType tag);
    __aicore__ inline TTagType GetUserTag() const;
    // symbol override
    __aicore__ inline void operator = (const SymbolOverrideAdd<T>& symbolOverride);
    __aicore__ inline void operator = (const SymbolOverrideSub<T>& symbolOverride);
    __aicore__ inline void operator = (const SymbolOverrideMul<T>& symbolOverride);
    __aicore__ inline void operator = (const SymbolOverrideDiv<T>& symbolOverride);
    __aicore__ inline void operator = (const SymbolOverrideOr<T>& symbolOverride);
    __aicore__ inline void operator = (const SymbolOverrideAnd<T>& symbolOverride);
    __aicore__ inline void operator = (const SymbolOverrideCompare<float>& symbolOverride);
    __aicore__ inline void operator = (const SymbolOverrideCompare<half>& symbolOverride);
    __aicore__ inline SymbolOverrideAdd<T> operator + (const LocalTensor<T>& src1Tensor) const;
    __aicore__ inline SymbolOverrideSub<T> operator - (const LocalTensor<T>& src1Tensor) const;
    __aicore__ inline SymbolOverrideMul<T> operator *(const LocalTensor<T>& src1Tensor) const;
    __aicore__ inline SymbolOverrideDiv<T> operator / (const LocalTensor<T>& src1Tensor) const;
    __aicore__ inline SymbolOverrideOr<T> operator | (const LocalTensor<T>& src1Tensor) const;
    __aicore__ inline SymbolOverrideAnd<T> operator & (const LocalTensor<T>& src1Tensor) const;
    __aicore__ inline SymbolOverrideCompare<T> operator < (const LocalTensor<T>& src1Tensor) const;
    __aicore__ inline SymbolOverrideCompare<T> operator > (const LocalTensor<T>& src1Tensor) const;
    __aicore__ inline SymbolOverrideCompare<T> operator != (const LocalTensor<T>& src1Tensor) const;
    __aicore__ inline SymbolOverrideCompare<T> operator == (const LocalTensor<T>& src1Tensor) const;
    __aicore__ inline SymbolOverrideCompare<T> operator <= (const LocalTensor<T>& src1Tensor) const;
    __aicore__ inline SymbolOverrideCompare<T> operator >= (const LocalTensor<T>& src1Tensor) const;
    __aicore__ inline void SetShapeInfo(const ShapeInfo& shapeInfo);
    __aicore__ inline ShapeInfo GetShapeInfo() const;

public:
#ifndef __DAV_C220_CUBE__
    typename ShapeInfoParams<T, PrimType>::Params shapeInfo_;
#endif
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    std::ostringstream os_;
#endif

private:
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    inline void PrintTypicalFloat(uint32_t len, uint32_t dataSize);
#endif
};

template <typename T> class GlobalTensor : public BaseGlobalTensor<T> {
public:
    using PrimType = PrimT<T>;
    __aicore__ inline GlobalTensor<T>() {}
    __aicore__ inline void SetGlobalBuffer(__gm__ PrimType* buffer, uint64_t bufferSize);
    __aicore__ inline void SetGlobalBuffer(__gm__ PrimType* buffer);
    __aicore__ inline const __gm__ PrimType* GetPhyAddr() const;
    __aicore__ inline __gm__ PrimType* GetPhyAddr(const uint64_t offset) const;
    __aicore__ inline __inout_pipe__(S) PrimType GetValue(const uint64_t offset) const;
    __aicore__ inline __inout_pipe__(S) __gm__ PrimType& operator()(const uint64_t offset) const;
    __aicore__ inline void SetValue(const uint64_t offset, PrimType value);

    __aicore__ inline uint64_t GetSize() const;
    __aicore__ inline GlobalTensor operator[](const uint64_t offset) const;
    __aicore__ inline void SetShapeInfo(const ShapeInfo& shapeInfo);
    __aicore__ inline ShapeInfo GetShapeInfo() const;
    template<CacheRwMode rwMode = CacheRwMode::RW>
    __aicore__ inline void SetL2CacheHint(CacheMode mode);

public:
    // element number of Tensor
    uint64_t bufferSize_;
#ifndef __DAV_C220_CUBE__
    typename ShapeInfoParams<T, PrimType>::Params shapeInfo_;
#endif
    CacheMode cacheMode_ = CacheMode::CACHE_MODE_NORMAL;
};
} // namespace AscendC
#endif // KERNEL_TENSOR_H
