/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_tensor_info.h
 * \brief matmul input tensor manager
 */

#ifndef IMPL_MATMUL_PARAM_MATMUL_TENSOR_INFO_H
#define IMPL_MATMUL_PARAM_MATMUL_TENSOR_INFO_H

#include "../utils/matmul_module.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, const auto &MM_CFG, class INPUT_TYPE, typename = void>
class MatmulTensorInfo {
    using SrcT = typename INPUT_TYPE::T;

    MATMUL_USE_MODULE(MatmulShapeInfo);
public:
    __aicore__ inline MatmulTensorInfo() = default;
    __aicore__ inline ~MatmulTensorInfo() = default;
    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline GlobalTensor<SrcT> GetGlobalTensor() const
    {
        GlobalTensor<SrcT> globalMatrix;
        globalMatrix.SetGlobalBuffer(globalMatrix_);
        return globalMatrix;
    }

    __aicore__ inline LocalTensor<SrcT> GetLocalTensor() const
    {
        LocalTensor<SrcT> localMatrix;
        localMatrix.SetAddr(localMatrix_.address_);
        return localMatrix;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void SetGlobalTensor(const GlobalTensor<SrcT>& globalMatrix, bool isTranspose)
    {
        globalMatrix_ = globalMatrix.address_;
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            CheckMatrixA(isTranspose);
            MATMUL_MODULE(MatmulShapeInfo)->SetTransposeA(isTranspose);
        } else {
            CheckMatrixB(isTranspose);
            MATMUL_MODULE(MatmulShapeInfo)->SetTransposeB(isTranspose);
        }
    }

    __aicore__ inline void SetLocalTensor(const LocalTensor<SrcT>& localMatrix, bool isTranspose)
    {
        localMatrix_.address_ = localMatrix.address_;
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            CheckMatrixA(isTranspose);
            CheckMatrixAFromLocalMemory();
            MATMUL_MODULE(MatmulShapeInfo)->SetTransposeA(isTranspose);
        } else {
            CheckMatrixB(isTranspose);
            CheckMatrixBFromLocalMemory();
            MATMUL_MODULE(MatmulShapeInfo)->SetTransposeB(isTranspose);
        }
    }

    template <bool isTrans = false>
    __aicore__ inline int GetBaseUseHeight() const
    {
        if constexpr (isTrans) {
            return MATMUL_CONST_PARAM_VAR.baseUseK_;
        } else {
            return MATMUL_CONST_PARAM_VAR.baseUseM_;
        }
    }

private:
    __aicore__ inline void CheckMatrixA(bool isTransposeA)
    {
        ASCENDC_ASSERT((isTransposeA <= INPUT_TYPE::isTrans), {
            KERNEL_LOG(KERNEL_ERROR, "It is not allowed to set matrix A transpose when matmul A transpose is not defined.");
        });
#if __CCE_AICORE__ == 220
        if constexpr (IsSameType<SrcT, int4b_t>::value) {
            ASCENDC_ASSERT(!isTransposeA, { KERNEL_LOG(KERNEL_ERROR,
                "When matrix A DType is int4, matrix A should not be transposed");});
        }
#elif __CCE_AICORE__ == 200
        if constexpr (IsSameType<SrcT, int8_t>::value) {
            ASCENDC_ASSERT(!isTransposeA, { KERNEL_LOG(KERNEL_ERROR,
                "When matrix A DType is int8, matrix A should not be transposed");});
        }
#endif
    }

    __aicore__ inline void CheckMatrixAFromLocalMemory()
    {
        // A/B does not come from GM with IBShare is not support
        if constexpr (DoMatmulIBShareNorm(MM_CFG) && INPUT_TYPE::ibShare) {
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR, "It is not allowed to set matrix A whose src::pos is L1 when matmul A is ibShare.");
            });
        }
    }

    __aicore__ inline void CheckMatrixB(bool isTransposeB)
    {
        ASCENDC_ASSERT((isTransposeB <= INPUT_TYPE::isTrans), {
            KERNEL_LOG(KERNEL_ERROR, "It is not allowed to set matrix B transpose when matmul B transpose is not defined.");
        });
    }

    __aicore__ inline void CheckMatrixBFromLocalMemory()
    {
        // A/B does not come from GM with IBShare is not support
        if constexpr (DoMatmulIBShareNorm(MM_CFG) && INPUT_TYPE::ibShare) {
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR, "It is not allowed to set matrix B whose src::pos is L1 when matmul B is ibShare.");
            });
        }
    }

    LocalTensor<TensorTrait<SrcT>> localMatrix_;
    __gm__ SrcT* globalMatrix_;
};

template <typename IMPL, const auto &MM_CFG, class INPUT_TYPE>
class MatmulTensorInfo<IMPL, MM_CFG, INPUT_TYPE, enable_if_t<IsIntrablock<MM_CFG>>> {
    using SrcT = typename INPUT_TYPE::T;

    MATMUL_USE_MODULE(MatmulShapeInfo);
public:
    __aicore__ inline MatmulTensorInfo() = default;
    __aicore__ inline ~MatmulTensorInfo() = default;
    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline GlobalTensor<SrcT> GetGlobalTensor() const
    {
        GlobalTensor<SrcT> globalMatrix;
        if constexpr (IS_INTRA_BLOCK) {
            globalMatrix.SetGlobalBuffer(intrablockGlobalMatrix_);
        } else {
            globalMatrix.SetGlobalBuffer(globalMatrix_);
        }
        return globalMatrix;
    }

    __aicore__ inline LocalTensor<SrcT> GetLocalTensor() const
    {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Intrablock only support inputs from GM."); });
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void SetGlobalTensor(const GlobalTensor<SrcT>& globalMatrix, bool isTranspose)
    {
        if constexpr (IS_INTRA_BLOCK) {
            intrablockGlobalMatrix_ = globalMatrix.address_;
        } else {
            globalMatrix_ = globalMatrix.address_;
        }
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            MATMUL_MODULE(MatmulShapeInfo)->SetTransposeA(isTranspose);
        } else {
            MATMUL_MODULE(MatmulShapeInfo)->SetTransposeB(isTranspose);
        }
    }

    __aicore__ inline void SetLocalTensor(const LocalTensor<SrcT>& localMatrix, bool isTranspose)
    {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Intrablock only support inputs from GM."); });
    }

private:
    __gm__ SrcT* globalMatrix_;
    __gm__ SrcT* intrablockGlobalMatrix_;
};

template <const auto& MM_CFG, typename INPUT_TYPE>
constexpr bool IsSparseMatmul = (INPUT_TYPE::TAG == InputTypeTag::B) &&
    HasSparseIndex<INPUT_TYPE>() && DoMatmulMDL(MM_CFG);

template <typename IMPL, const auto &MM_CFG, class INPUT_TYPE>
class MatmulTensorInfo<IMPL, MM_CFG, INPUT_TYPE, enable_if_t<IsSparseMatmul<MM_CFG, INPUT_TYPE>>> {
    using SrcT = typename INPUT_TYPE::T;

    MATMUL_USE_MODULE(MatmulShapeInfo);
public:
    __aicore__ inline MatmulTensorInfo() = default;
    __aicore__ inline ~MatmulTensorInfo() = default;
    __aicore__ inline LocalTensor<SrcT> GetLocalTensor() const
    {
        LocalTensor<SrcT> localMatrix;
        localMatrix.SetAddr(localMatrix_.address_);
        return localMatrix;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline GlobalTensor<SrcT> GetGlobalTensor() const
    {
        GlobalTensor<SrcT> globalMatrix;
        globalMatrix.SetGlobalBuffer(globalMatrix_);
        return globalMatrix;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void SetGlobalTensor(const GlobalTensor<SrcT>& globalMatrix, bool isTranspose)
    {
        globalMatrix_ = globalMatrix.address_;
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeB(isTranspose);
    }

    __aicore__ inline void SetLocalTensor(const LocalTensor<SrcT>& localMatrix, bool isTranspose)
    {
        localMatrix_.address_ = localMatrix.address_;
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeB(isTranspose);
    }

    __aicore__ inline void SetGlobalSparseIndex(const GlobalTensor<uint8_t>& indexGlobal)
    {
        indexGlobal_ = indexGlobal;
    }

    __aicore__ inline void SetLocalSparseIndex(const LocalTensor<uint8_t>& indexLocal)
    {
        indexLocal_ = indexLocal;
    }

    __aicore__ inline GlobalTensor<uint8_t> GetGlobalSparseIndex()
    {
        return indexGlobal_;
    }

    __aicore__ inline LocalTensor<uint8_t> GetLocalSparseIndex()
    {
        return indexLocal_;
    }

    template <bool IS_TRANS = false>
    __aicore__ inline int GetBaseUseHeight() const
    {
        if constexpr (IS_TRANS) {
            return MATMUL_CONST_PARAM_VAR.baseUseN_;
        } else {
            return MATMUL_CONST_PARAM_VAR.baseUseK_;
        }
    }

private:
    LocalTensor<TensorTrait<SrcT>> localMatrix_;
    __gm__ SrcT* globalMatrix_;
    GlobalTensor<uint8_t> indexGlobal_;
    LocalTensor<uint8_t> indexLocal_;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_PARAM_MATMUL_TENSOR_INFO_H
