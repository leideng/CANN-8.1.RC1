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
 * \file matmul_server_aux.h
 * \brief
 */
#ifndef IMPL_MATMUL_KFC_MATMUL_SERVER_AUX_H
#define IMPL_MATMUL_KFC_MATMUL_SERVER_AUX_H

#include "matmul_server_impl.h"

namespace AscendC {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE = C_TYPE,
    const auto& MM_CFG = CFG_NORM, class MM_CB = AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
    MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
struct MatmulInstBase {
    __aicore__ inline MatmulInstBase(){};
};
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
struct MatmulInstShared : MatmulInstBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY> {
    __aicore__ inline MatmulInstShared(){};
    AscendC::MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY> cubeObj[1];
};
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
struct MatmulInst : MatmulInstBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY> {
    __aicore__ inline MatmulInst(){};
    AscendC::MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY> cubeObj[MIX_NUM];
};

template <bool SHARED, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
    class MM_CB, MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
struct MatmulInstAux {
    __aicore__ inline MatmulInstAux(){};
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
struct MatmulInstAux<true, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY> {
    __aicore__ inline MatmulInstAux(){};
    using MATMUL = MatmulInstShared<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
struct MatmulInstAux<false, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY> {
    __aicore__ inline MatmulInstAux(){};
    using MATMUL = MatmulInst<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE = C_TYPE, const auto& MM_CFG = CFG_NORM,
class MM_CB = AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulServiceAuxBase {
    using SrcT = typename A_TYPE::T;
    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    using DstT = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    template <class... Args> friend struct AscendC::GetCubeObjConfig;
    constexpr static bool enableMixDualMaster = ToMatmulConfig(MM_CFG).enableMixDualMaster;
    constexpr static bool enableABShare = A_TYPE::ibShare && B_TYPE::ibShare;

public:
    __aicore__ inline MatmulServiceAuxBase() {}
    typename MatmulInstAux<IsSharedMatmul<MM_CFG>(), A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB,
        MATMUL_POLICY>::MATMUL cubeObj;

    // stub functions for MatmulImplBase
    __aicore__ inline void Init(TCubeTiling *cubeTiling, TPipe *tpipe = nullptr)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.Init(cubeTiling, tpipe);
        }
    }

    __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgK)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetOrgShape(orgM, orgN, orgK);
        }
    }
    __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgKa, int orgKb, int orgKc = 0)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc);
        }
    }
    __aicore__ inline void SetSingleShape(int singleM, int singleN, int singleK)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetSingleShape(singleM, singleN, singleK);
        }
    }
    __aicore__ inline void SetTail(int tailM = -1, int tailN = -1, int tailK = -1)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetTail(tailM, tailN, tailK);
        }
    }

    __aicore__ inline void SetTensorA(const GlobalTensor<SrcAT>& gm, bool isTransposeA = false)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetTensorA(gm, isTransposeA);
        }
    }

    __aicore__ inline void SetTensorAWithCopy(const GlobalTensor<SrcAT>& gm, const LocalTensor<SrcAT>& leftMatrix,
        bool isTransposeA = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "SetTensorAWithCopy not support when enableMixDualMaster is enabled");
    }
    __aicore__ inline void SetTensorB(const GlobalTensor<SrcBT>& gm, bool isTransposeB = false)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetTensorB(gm, isTransposeB);
        }
    }

    __aicore__ inline void SetTensorBWithCopy(const GlobalTensor<SrcBT>& gm, const LocalTensor<SrcBT>& rightMatrix,
        bool isTransposeB = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "SetTensorBWithCopy not support when enableMixDualMaster is enabled");
    }
    __aicore__ inline void SetBias(const GlobalTensor<BiasT>& biasGlobal)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetBias(biasGlobal);
        }
    }
    __aicore__ inline void SetTensorA(const LocalTensor<SrcAT>& leftMatrix, bool isTransposeA = false)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if (__CCE_AICORE__ == 220)
            ASSERT("SetTensorA localTensor not support when enableMixDualMaster is enabled");
#endif
        }   
    }
    __aicore__ inline void SetTensorB(const LocalTensor<SrcBT>& rightMatrix, bool isTransposeB = false)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if (__CCE_AICORE__ == 220)
            ASSERT("SetTensorB localTensor not support when enableMixDualMaster is enabled");
#endif
        } 
    }
    __aicore__ inline void SetBias(const LocalTensor<BiasT>& inputBias)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if (__CCE_AICORE__ == 220)
            ASSERT("SetBias localTensor not support when enableMixDualMaster is enabled");
#endif
        } 
    }
    __aicore__ inline void SetTensorA(SrcAT aScalar)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetTensorA(aScalar);
        }
    }
    __aicore__ inline void SetTensorB(SrcBT bScalar)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetTensorB(bScalar);
        }
    }
    __aicore__ inline void DisableBias()
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.DisableBias();
        }
    }
    __aicore__ inline void ClearBias()
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.ClearBias();
        }
    }
    __aicore__ inline void SetSelfDefineData(const uint64_t dataPtr)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetSelfDefineData(dataPtr);
        }
    }
    __aicore__ inline void SetUserDefInfo(const uint64_t tilingPtr)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetUserDefInfo(tilingPtr);
        }
    }
    __aicore__ inline void SetQuantScalar(const uint64_t quantScalar)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetQuantScalar(quantScalar);
        }
    }
    __aicore__ inline void SetQuantVector(const GlobalTensor<uint64_t>& quantTensor)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetQuantVector(quantTensor);
        }
    }
    template <class T> __aicore__ inline void SetWorkspace(__gm__ T* addr, int size)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "SetWorkspace not support when enableMixDualMaster is enabled");
    }
    template <class T> __aicore__ inline void SetWorkspace(GlobalTensor<T>& addr)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "SetWorkspace not support when enableMixDualMaster is enabled");
    }
    __aicore__ inline void End(){};
    __aicore__ inline void SetHF32(bool enableHF32 = false, int32_t transMode = 0)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            cubeObj.cubeObj[0].mul.SetHF32(enableHF32, transMode);
        }
    }

    template <bool sync = true> __aicore__ inline bool Iterate(bool enPartialSum = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "Iterate not support when enableMixDualMaster is enabled");
        return false;
    };
    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0,
        bool enSequentialWrite = false, bool waitIterateAll = false, bool fakeMsg = false)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            constexpr uint16_t eventID = 9U;
            WaitEvent(eventID);
            cubeObj.cubeObj[0].mul.IterateAll(gm, enAtomic, enSequentialWrite, waitIterateAll, fakeMsg);
            if (sync || waitIterateAll){
                NotifyEvent<PIPE_FIX>(cubeObj.cubeObj[0].instID);
            }
            cubeObj.cubeObj[0].mul.End();
        }
    }
    template <bool sync = true>
    __aicore__ inline void IterateAll(const LocalTensor<DstT>& ubCmatrix, uint8_t enAtomic = 0)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if (__CCE_AICORE__ == 220)
            ASSERT("IterateAll localTensor not support when enableMixDualMaster is enabled");
#endif
        }
    }
    __aicore__ inline void WaitIterateAll() {};
    template <bool sync = true, bool doPad = false>
    __aicore__ inline void GetTensorC(const LocalTensor<DstT>& c, uint8_t enAtomic = 0,
        bool enSequentialWrite = false, uint32_t height = 0, uint32_t width = 0, uint32_t srcGap = 0, uint32_t dstGap = 0)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "GetTensorC not support when enableMixDualMaster is enabled");
    }
    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0,
        bool enSequentialWrite = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "GetTensorC not support when enableMixDualMaster is enabled");
    }
    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& cLocal,
        uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "GetTensorC not support when enableMixDualMaster is enabled");
    }
    template <bool sync = true>
    __aicore__ inline GlobalTensor<DstT> GetTensorC(uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "GetTensorC not support when enableMixDualMaster is enabled");
        GlobalTensor<DstT> global;
        return global;
    };
    template <bool sync = true, bool waitIterateBatch = false>
    __aicore__ inline void IterateBatch(const GlobalTensor<DstT>& gm, uint32_t batchA, uint32_t batchB,
        bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0,
        const uint32_t matrixStrideC = 0)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "IterateBatch not support when enableMixDualMaster is enabled");
    }
    template <bool sync = true>
    __aicore__ inline void IterateBatch(const LocalTensor<DstT>& ubCmatrix, uint32_t batchA, uint32_t batchB,
        bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0,
        const uint32_t matrixStrideC = 0)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "IterateBatch not support when enableMixDualMaster is enabled");
    }
    template <bool sync = true, bool waitIterateBatch = false>
    __aicore__ inline void IterateNBatch(const uint32_t batchLoop, uint32_t batchA, uint32_t batchB,
        bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0,
        const uint32_t matrixStrideC = 0)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "IterateNBatch not support when enableMixDualMaster is enabled");
    }
    template <bool sync = true>
    __aicore__ inline GlobalTensor<DstT> GetBatchTensorC(uint32_t batchA, uint32_t batchB,
        bool enSequentialWrite = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "GetBatchTensorC not support when enableMixDualMaster is enabled");
    }
    template <bool sync = true>
    __aicore__ inline GlobalTensor<DstT> GetBatchC(uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "GetBatchC not support when enableMixDualMaster is enabled");
    }
    template <bool sync = true, bool doPad = false>
    __aicore__ inline void GetBatchTensorC(const LocalTensor<DstT>& c, uint32_t batchA, uint32_t batchB,
        bool enSequentialWrite = false, uint32_t height = 0, uint32_t width = 0, uint32_t srcGap = 0,
        uint32_t dstGap = 0)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "GetBatchTensorC not support when enableMixDualMaster is enabled");
    }
    template <bool sync = true, bool doPad = false>
    __aicore__ inline void GetBatchC(const LocalTensor<DstT>& c, uint32_t batchA, uint32_t batchB,
        bool enSequentialWrite = false, uint32_t height = 0, uint32_t width = 0, uint32_t srcGap = 0,
        uint32_t dstGap = 0)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "GetBatchC not support when enableMixDualMaster is enabled");
    }
    __aicore__ inline void WaitIterateBatch()
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "WaitIterateBatch not support when enableMixDualMaster is enabled");
    }
    __aicore__ inline void SetLocalWorkspace(const LocalTensor<uint8_t>& tmpBuffer)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "SetLocalWorkspace not support when enableMixDualMaster is enabled");
    }
    __aicore__ inline void AsyncGetTensorC(const LocalTensor<DstT>& c)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "AsyncGetTensorC not support when enableMixDualMaster is enabled");
    }
    __aicore__ inline void WaitGetTensorC()
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster &&
            "WaitGetTensorC not support when enableMixDualMaster is enabled");
    }
    template <bool isTurnOnDebug = true>
    __aicore__ inline MatrixOffset GetOffsetC()
    {
        if constexpr (isTurnOnDebug) {
            static_assert(!isTurnOnDebug, "Debug is not supported!");
        }
    }
};

// Match Policy with CallBack paramter
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
class MatmulServiceAux
: public MatmulServiceAuxBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY> {
public:
    __aicore__ inline MatmulServiceAux() {}
};
} // namespace AscendC
#endif // __MATMUL_SERVER_H__