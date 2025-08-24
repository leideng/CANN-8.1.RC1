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
 * \file matmul.h
 * \brief
 */
#ifndef LIB_MATMUL_MATMUL_H
#define LIB_MATMUL_MATMUL_H

#include <type_traits>
#include "lib/matmul/constant_tiling.h"
#include "lib/matmul/tiling.h"
#include "../../impl/matmul/policy/matmul_policy.h"
#include "../../impl/matmul/utils/matmul_call_back.h"
#include "../../impl/matmul/utils/matmul_module.h"

namespace AscendC {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG>
struct MatmulApiConfig {
    using AType = A_TYPE;
    using BType = B_TYPE;
    using CType = C_TYPE;
    using BiasType = BIAS_TYPE;
    constexpr static MatmulConfig Config = ToMatmulConfig(MM_CFG);
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE = C_TYPE, const auto& MM_CFG = CFG_NORM,
class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy), typename = void>
class MatmulImpl
{
public:
    using AType = A_TYPE;
    using BType = B_TYPE;
    using CType = C_TYPE;
    using BiasType = BIAS_TYPE;
private:
    using L0cT = typename GetDstType<typename A_TYPE::T>::Type;
    using SrcT = typename A_TYPE::T;
    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    using DstT = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

public:
    __aicore__ inline MatmulImpl() {}
    __aicore__ inline void Init(const TCubeTiling* __restrict cubeTiling, TPipe* tpipe = nullptr) {}
    __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgK) {}
    __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgKa, int orgKb, int orgKc = 0) {}
    __aicore__ inline void SetSingleShape(int singleM, int singleN, int singleK) {}
    __aicore__ inline void SetTail(int tailM = -1, int tailN = -1, int tailK = -1) {}
    __aicore__ inline void SetTensorA(const GlobalTensor<SrcAT>& gm, bool isTransposeA = false) {}
    __aicore__ inline void SetTensorB(const GlobalTensor<SrcBT>& gm, bool isTransposeB = false) {}
    __aicore__ inline void SetBias(const GlobalTensor<BiasT>& biasGlobal) {}
    __aicore__ inline void SetSelfDefineData(const uint64_t dataPtr) {}
    __aicore__ inline void SetUserDefInfo(const uint64_t tilingPtr) {}
    __aicore__ inline void SetSparseIndex(const GlobalTensor<uint8_t>& indexGlobal);
    __aicore__ inline void SetAntiQuantScalar(const SrcT offsetScalar, const SrcT scaleScalar) {}
    __aicore__ inline void SetAntiQuantVector(const LocalTensor<SrcT> &offsetTensor,
        const LocalTensor<SrcT> &scaleTensor) {}
    __aicore__ inline void SetQuantScalar(const uint64_t quantScalar) {}
    __aicore__ inline void SetQuantVector(const GlobalTensor<uint64_t>& quantTensor) {}
    __aicore__ inline void SetTensorA(const LocalTensor<SrcAT>& leftMatrix, bool isTransposeA = false) {}
    __aicore__ inline void SetTensorAWithCopy(const GlobalTensor<SrcAT>& gm, const LocalTensor<SrcAT>& leftMatrix,
        bool isTransposeA = false) {}
    __aicore__ inline void SetTensorB(const LocalTensor<SrcBT>& rightMatrix, bool isTransposeB = false) {}
    __aicore__ inline void SetTensorA(SrcAT aScalar) {}
    __aicore__ inline void SetTensorB(SrcBT bScalar) {}
    __aicore__ inline void SetTensorBWithCopy(const GlobalTensor<SrcBT>& gm, const LocalTensor<SrcBT>& rightMatrix,
        bool isTransposeB = false) {}
    __aicore__ inline void SetBias(const LocalTensor<BiasT>& inputBias) {}
    __aicore__ inline void SetBatchNum(int32_t batchA, int32_t batchB) {}
    __aicore__ inline void DisableBias() {}
    __aicore__ inline void ClearBias() {}
    template <bool sync = true> __aicore__ inline bool Iterate(bool enPartialSum = false) {}
    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0,
        bool enSequentialWrite = false, bool waitIterateAll = false, bool fakeMsg = false) {}
    template <bool sync = true>
    __aicore__ inline void IterateAll(const LocalTensor<DstT>& ubCmatrix, uint8_t enAtomic = 0) {}

    __aicore__ inline void IterateBatch(const GlobalTensor<DstT>& gm,
        bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite, const uint32_t matrixStrideA = 0,
        const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0) {}
    __aicore__ inline void IterateBatch(const LocalTensor<DstT>& ubCmatrix,
        bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite, const uint32_t matrixStrideA = 0,
        const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0) {}

    template <bool sync = true>
    __aicore__ inline void GetTensorC(const LocalTensor<DstT>& co2Local, uint8_t enAtomic = 0,
        bool enSequentialWrite = false) {}
    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0,
        bool enSequentialWrite = false) {}
    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<DstT> &gm, const LocalTensor<DstT> &co2Local,
        uint8_t enAtomic = 0, bool enSequentialWrite = false) {}
    template <bool isTurnOnDebug = true>
    __aicore__ inline MatrixOffset GetOffsetC() {}
    __aicore__ inline void End() {}
    __aicore__ inline void SetHF32(bool enableHF32 = false, int32_t transMode = 0) {}
    __aicore__ inline void SetSubBlockIdx(uint8_t subBlockIdx) {}
    __aicore__ inline uint8_t GetSubBlockIdx() {}
    template <class T> __aicore__ inline void SetWorkspace(__gm__ const T* addr, int size) {}
    template <class T> __aicore__ inline void SetWorkspace(GlobalTensor<T>& addr) {}

    __aicore__ inline void SetLocalWorkspace(const LocalTensor<uint8_t>& tmpBuffer) {}
    using CallBack = MM_CB;
};

} // namespace AscendC
// Compatible with the previously used matmul namespace
namespace matmul = AscendC;
#include "../../impl/matmul/matmul_impl_base.h"
#include "../../impl/matmul/matmul_impl.h"
#include "../../impl/matmul/batch_matmul_impl.h"
#endif
