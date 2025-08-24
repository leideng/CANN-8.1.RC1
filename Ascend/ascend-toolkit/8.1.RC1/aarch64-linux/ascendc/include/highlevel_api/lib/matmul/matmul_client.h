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
 * \file matmul_client.h
 * \brief
 */
#ifndef LIB_MATMUL_MATMUL_CLIENT_H
#define LIB_MATMUL_MATMUL_CLIENT_H

#include "kernel_operator.h"
#include "lib/matmul/constant_tiling.h"
#include "lib/matmul/tiling.h"
#include "../../impl/matmul/policy/matmul_policy.h"
#include "../../impl/matmul/utils/matmul_call_back.h"
#include "../../impl/matmul/utils/matmul_module.h"
#include "../../impl/matmul/utils/matmul_utils.h"
#if ASCENDC_CPU_DEBUG
#include "../../impl/matmul/kfc/matmul_server_aux.h"
#endif

namespace AscendC {

constexpr int32_t VECTOR_QUANT_MODE = 2;

// Service function of the Matmul on the AIV client side, which is the unit for sending messages.
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG = CFG_NORM,
    class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulClientBase {
    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    using DstT = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

public:
    __aicore__ inline void Init(const TCubeTiling* __restrict cubeTiling, TPipe* tpipe = nullptr)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.Init(cubeTiling, tpipe);
            }
#endif
            return;
        }
        ASSERT(sizeof(KfcMsg) % CACHE_LINE_SIZE == 0);
        ASSERT(cubeTiling != nullptr && "cubeTiling cannot be nullptr when init matmul client");
        ASSERT(sizeof(TCubeTiling) % sizeof(uint64_t) == 0);
        if constexpr (A_TYPE::ibShare && B_TYPE::ibShare) {
            if (GetSubBlockIdxImpl() == 1) {
                return;
            }
        }
        constexpr uint32_t tCubeTilingSize = ConstCeil(sizeof(TCubeTiling), CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
        int32_t ubAddr = -1;
        GM_ADDR tilingGM = client->AllocUB(tCubeTilingSize, ubAddr);
        auto tempTilingGM = reinterpret_cast<__gm__ uint32_t*>(tilingGM);
        auto tempTiling = reinterpret_cast<uint32_t*>(const_cast<TCubeTiling*> (cubeTiling));
        for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); ++i, ++tempTilingGM, ++tempTiling) {
            *tempTilingGM = *tempTiling;
        }
        this->cubeTiling.SetTiling(cubeTiling);
        GlobalTensor<int64_t> global;
        for (int i = 0; i < tCubeTilingSize; i += CACHE_LINE_SIZE) {
            Barrier();
            global.SetGlobalBuffer((__gm__ int64_t*)(tilingGM + i));
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
        }
        Barrier();

        auto msg = client->AllocMessage();
        client->ubMsg->tilingInfo.tilingAddr = tilingGM;
        client->ubMsg->head = KfcMsgMakeFlag(KFC_Enum::MMFUN_INIT, this->instIdx);
        client->ubMsg->ubAddr = ubAddr;
        client->PostMessage<false>(msg); // Initialize the local client after the expected processing is complete.

        *((uint64_t*)&kfcMsg_) = 0;
        *((uint64_t*)&(kfcMsg_.body)) = 0;
        nIter_ = ConstCeil(this->cubeTiling.GetSingleCoreN(), this->cubeTiling.GetBaseN());
        mIter_ = ConstCeil(this->cubeTiling.GetSingleCoreM(), this->cubeTiling.GetBaseM());
        mnIter_ = nIter_ * mIter_;
        cacheWorkspaceAddr = nullptr;
    }

    template <class T> __aicore__ inline void SetWorkspace(GlobalTensor<T>& addr)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "SetWorkspace not support when enableMixDualMaster is enabled");
        ASSERT(addr.GetSize() > 0);
        SetWorkspace(addr.GetPhyAddr(), addr.GetSize() * sizeof(T));
    }
    template <class T> __aicore__ inline void SetWorkspace(__gm__ const T* addr, int size)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "SetWorkspace not support when enableMixDualMaster is enabled");
        ASSERT(addr != nullptr);
        if constexpr (ToMatmulConfig(MM_CFG).singleCoreM == 0) {
            ASSERT(!this->cubeTiling.IsNull());
        }

        uint64_t offset = mnIter_ * cubeTiling.GetBaseN() * cubeTiling.GetBaseM() * sizeof(DstT);
        cacheWorkspaceAddr = reinterpret_cast<GM_ADDR>(const_cast<__gm__ T*>(addr));
        cOffset_ = 0;
    }

    __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgK)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetOrgShape(orgM, orgN, orgK);
            }
#endif
            return;
        }
        SetOrgShape(orgM, orgN, orgK, orgK, orgN);
    }

    __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgKa, int orgKb, int orgKc = 0)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc);
            }
#endif
            return;
        }
        kfcMsg_.orgShape.orgM = orgM;
        kfcMsg_.orgShape.orgN = orgN;
        kfcMsg_.orgShape.orgKa = orgKa;
        kfcMsg_.orgShape.orgKb = orgKb;
        kfcMsg_.orgShape.orgKc = orgKc;

        PostMessage<KFC_Enum::MMFUN_SET_ORG_SHAPE, false>();
    }

    __aicore__ inline void SetSingleShape(int singleM, int singleN, int singleK)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetSingleShape(singleM, singleN, singleK);
            }
#endif
            return;
        }
        SetTail(singleM, singleN, singleK);
    }

    __aicore__ inline void SetTail(int tailM = -1, int tailN = -1, int tailK = -1)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetTail(tailM, tailN, tailK);
            }
#endif
            return;
        }
        if (tailM != -1) {
            mIter_ = ConstCeil(tailM, cubeTiling.GetBaseM());
        }
        if (tailN != -1) {
            nIter_ = ConstCeil(tailN, cubeTiling.GetBaseN());
        }
        mnIter_ = nIter_ * mIter_;

        kfcMsg_.body.singleM = tailM;
        kfcMsg_.body.singleN = tailN;
        kfcMsg_.body.singleK = tailK;
        kfcMsg_.body.setTail = 1;
    }

    // transMode only support 0 or 1
    // 0: round mode is round to the nearest tie to even
    // 1: round mode is round to the nearest tie away from zero
    __aicore__ inline void SetHF32(bool enableHF32 = false, int32_t transMode = 0)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetHF32(enableHF32, transMode);
            }
#endif
            return;
        }
        kfcMsg_.body.enHF32 = enableHF32;
        kfcMsg_.body.hf32TransMode = transMode;

        PostMessage<KFC_Enum::MMFUN_SET_HF32, false>();
    }

    __aicore__ inline void SetTensorA(const LocalTensor<SrcAT>& leftMatrix, bool isTransposeA = false)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if (__CCE_AICORE__ == 220)
            ASSERT("SetTensorA localTensor not support when enableMixDualMaster is enabled");
#endif
            return;
        }
        ASSERT(isTransposeA <= A_TYPE::isTrans &&
            "It is not allowed to do A transpose when matmul A transpose is not defined.");
        kfcMsg_.body.isTransA = static_cast<uint32_t>(isTransposeA);
        kfcMsg_.body.setTensorA = 1;
        kfcMsg_.body.isFirstIter = 1;
        if constexpr (A_TYPE::pos == TPosition::TSCM) {
            kfcMsg_.body.aAddr = GetTscmAddr(leftMatrix);
            kfcMsg_.body.sizeAmatrix = leftMatrix.GetSize() * sizeof(SrcAT);
        } else {
            kfcMsg_.body.aAddr = GetGlobalAddr<SrcAT, true>(leftMatrix);
            kfcMsg_.body.sizeAmatrix = leftMatrix.GetSize() * sizeof(SrcAT);
        }
    }

    __aicore__ inline void SetTensorAWithCopy(const GlobalTensor<SrcAT>& gm, const LocalTensor<SrcAT>& leftMatrix,
        bool isTransposeA = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "SetTensorAWithCopy not support when enableMixDualMaster is enabled");
        ASSERT(A_TYPE::pos != TPosition::TSCM);
        kfcMsg_.body.isTransA = static_cast<uint32_t>(isTransposeA);
        kfcMsg_.body.setTensorA = 1;
        kfcMsg_.body.isFirstIter = 1;
        kfcMsg_.body.aAddr = GetGMAddrAndCopyUB(gm.GetPhyAddr(), leftMatrix);
        kfcMsg_.body.sizeAmatrix = leftMatrix.GetSize() * sizeof(SrcAT);
    }

    __aicore__ inline void SetTensorB(const LocalTensor<SrcBT>& rightMatrix, bool isTransposeB = false)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if (__CCE_AICORE__ == 220)
            ASSERT("SetTensorB localTensor not support when enableMixDualMaster is enabled");
#endif
            return;
        }
        ASSERT(isTransposeB <= B_TYPE::isTrans &&
            "It is not allowed to do B transpose when matmul B transpose is not defined.");
        kfcMsg_.body.isTransB = static_cast<uint32_t>(isTransposeB);
        kfcMsg_.body.setTensorB = 1;
        kfcMsg_.body.isFirstIter = 1;

        if constexpr (B_TYPE::pos == TPosition::TSCM) {
            kfcMsg_.body.bAddr = GetTscmAddr(rightMatrix);
            kfcMsg_.body.sizeBmatrix = rightMatrix.GetSize() * sizeof(SrcBT);
        } else {
            kfcMsg_.body.bAddr = GetGlobalAddr<SrcBT, true>(rightMatrix);
            kfcMsg_.body.sizeBmatrix = rightMatrix.GetSize() * sizeof(SrcBT);
        }
    }

    __aicore__ inline void SetTensorBWithCopy(const GlobalTensor<SrcBT>& gm, const LocalTensor<SrcBT>& rightMatrix,
        bool isTransposeB = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "SetTensorBWithCopy not support when enableMixDualMaster is enabled");
        ASSERT(A_TYPE::pos != TPosition::TSCM);
        kfcMsg_.body.isTransB = static_cast<uint32_t>(isTransposeB);
        kfcMsg_.body.setTensorB = 1;
        kfcMsg_.body.isFirstIter = 1;
        kfcMsg_.body.bAddr = GetGMAddrAndCopyUB(gm.GetPhyAddr(), rightMatrix);
        kfcMsg_.body.sizeBmatrix = rightMatrix.GetSize() * sizeof(SrcBT);
    }

    __aicore__ inline void SetBias(const LocalTensor<BiasT>& inputBias)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if (__CCE_AICORE__ == 220)
            ASSERT("SetBias localTensor not support when enableMixDualMaster is enabled");
#endif
            return;
        }
        kfcMsg_.body.setTensorBias = 1;
        if constexpr (BIAS_TYPE::pos == TPosition::TSCM) {
            kfcMsg_.body.biasAddr = GetTscmAddr(inputBias);
        } else {
            kfcMsg_.body.biasAddr = GetGlobalAddr<BiasT, true>(inputBias);
        }
    };

    __aicore__ inline void SetTensorA(const GlobalTensor<SrcAT>& gm, bool isTransposeA = false)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetTensorA(gm, isTransposeA);
            }
#endif
            return;
        }
        ASSERT(isTransposeA <= A_TYPE::isTrans &&
            "It is not allowed to do A transpose when matmul A transpose is not defined.");
        kfcMsg_.body.isTransA = static_cast<uint32_t>(isTransposeA);
        kfcMsg_.body.aAddr = reinterpret_cast<uint64_t>(gm.GetPhyAddr());
        kfcMsg_.body.sizeAmatrix = gm.GetSize() * sizeof(SrcAT);
        kfcMsg_.body.setTensorA = 1;
        kfcMsg_.body.isFirstIter = 1;
    }

    __aicore__ inline void SetTensorB(const GlobalTensor<SrcBT>& gm, bool isTransposeB = false)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetTensorB(gm, isTransposeB);
            }
#endif
            return;
        }
        ASSERT(isTransposeB <= B_TYPE::isTrans &&
            "It is not allowed to do B transpose when matmul B transpose is not defined.");
        kfcMsg_.body.isTransB = static_cast<uint32_t>(isTransposeB);
        kfcMsg_.body.bAddr = reinterpret_cast<uint64_t>(gm.GetPhyAddr());
        kfcMsg_.body.sizeBmatrix = gm.GetSize() * sizeof(SrcBT);
        kfcMsg_.body.setTensorB = 1;
        kfcMsg_.body.isFirstIter = 1;
    }

    __aicore__ inline void SetSelfDefineData(const uint64_t dataPtr)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetSelfDefineData(dataPtr);
            }
#endif
            return;
        }
        kfcMsg_.body.dataPtr = dataPtr;
    }

    __aicore__ inline void SetUserDefInfo(const uint64_t tilingPtr)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetUserDefInfo(tilingPtr);
            }
#endif
            return;
        }
        kfcMsg_.userDefInfo.tilingPtr = tilingPtr;
        PostMessage<KFC_Enum::MMFUN_SET_USER_DEF_INFO, false>();
    }

    __aicore__ inline void SetSparseIndex(const GlobalTensor<uint8_t>& indexGlobal)
    {
        ASSERT("SetSparseIndex is not supported in matmul client.");
        return;
    }

    __aicore__ inline void SetQuantScalar(const uint64_t quantScalar)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetQuantScalar(quantScalar);
            }
#endif
            return;
        }
        kfcMsg_.body.setQuant = 1;
        kfcMsg_.body.quantMode = 1;
        kfcMsg_.body.quantScalar = quantScalar;
    }

    __aicore__ inline void SetQuantVector(const GlobalTensor<uint64_t>& quantTensor)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetQuantVector(quantTensor);
            }
#endif
            return;
        }
        kfcMsg_.body.setQuant = 1;
        kfcMsg_.body.quantMode = VECTOR_QUANT_MODE;
        kfcMsg_.body.quantAddr = reinterpret_cast<uint64_t>(quantTensor.GetPhyAddr());
        kfcMsg_.body.quantSize = quantTensor.GetSize() * sizeof(uint64_t);
    }

    __aicore__ inline void SetBias(const GlobalTensor<BiasT>& biasGlobal)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetBias(biasGlobal);
            }
#endif
            return;
        }
        kfcMsg_.body.biasAddr = reinterpret_cast<uint64_t>(biasGlobal.GetPhyAddr());
        kfcMsg_.body.setTensorBias = 1;
    }

    __aicore__ inline void SetTensorA(SrcAT aScalar)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetTensorA(aScalar);
            }
#endif
            return;
        }
        auto temp1 = (uint8_t*)&(aScalar);
        auto temp2 = reinterpret_cast<uint8_t*>(&(kfcMsg_.body.aAddr));

        for (int i = 0; i < sizeof(SrcAT); i++, temp1++, temp2++) {
            *temp2 = *temp1;
        }
        kfcMsg_.body.setTensorA = 1;
    }

    __aicore__ inline void SetTensorB(SrcBT bScalar)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.SetTensorB(bScalar);
            }
#endif
            return;
        }
        auto temp1 = (uint8_t*)&(bScalar);
        auto temp2 = reinterpret_cast<uint8_t*>(&(kfcMsg_.body.aAddr));

        for (int i = 0; i < sizeof(SrcBT); i++, temp1++, temp2++) {
            *temp2 = *temp1;
        }
        kfcMsg_.body.setTensorB = 1;
    }

    __aicore__ inline void DisableBias()
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.DisableBias();
            }
#endif
            return;
        }
        kfcMsg_.body.setTensorBias = 0;
    }

    __aicore__ inline void ClearBias()
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                cubeObj.cubeObj[0].mul.ClearBias();
            }
#endif
            return;
        }
        DisableBias();
    }

    __aicore__ inline void End()
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            return;
        }
        if (isSyncGetC) {
            PostMessage<KFC_Enum::MMFUN_END, false>();
        }
    }

    template <bool sync = true> __aicore__ inline bool Iterate(bool enPartialSum = false)
    {
        TRACE_START(TraceId::KFC_CLIENT_POST_MSG);
        if (unlikely(kfcMsg_.body.isFirstIter)) {
            cntIter_ = 0;
            cOffset_ = 0;
            curProcess = 0;
            *((__gm__ uint64_t*)mmCntAddr_) = 0;
            GlobalTensor<uint64_t> global;
            global.SetGlobalBuffer((__gm__ uint64_t*)mmCntAddr_);
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
        } else {
            if (++cntIter_ >= mnIter_) {
                TRACE_STOP(TraceId::KFC_CLIENT_POST_MSG);
                return false;
            }
            if constexpr (!sync) {
                TRACE_STOP(TraceId::KFC_CLIENT_POST_MSG);
                return true;
            }
        }

        if constexpr (!sync) {  // Asynchronous mode. Only UB.
            ASSERT(cacheWorkspaceAddr != 0);  // The cache address must be configured in asynchronous mode.
            ASSERT(PhyPosIsUB(C_TYPE::pos));  // Asynchronous mode. Only UB.
        }

        isSyncGetC = sync;

        // Synchronous mode. no cache for the first time
        kfcMsg_.body.enPartialSum = enPartialSum;
        kfcMsg_.body.sync = sync;
        kfcMsg_.body.cAddr = reinterpret_cast<uint64_t>(cacheWorkspaceAddr);
        PostMessage<KFC_Enum::MMFUN_ITERATE, false>();
        SyncCubeWithVec<A_TYPE::ibShare, B_TYPE::ibShare>();
        TRACE_STOP(TraceId::KFC_CLIENT_POST_MSG);
        return true;
    }

    // Only support the mode that the IterateAll is asynchronous and GM output is continuous.
    // In discontinuous scenarios, the system stops responding.
    __aicore__ inline void WaitIterateAll()
    {
        ASSERT(!isSyncGetC); // Must be asynchronous mode
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                return;
            }
#endif
            WaitEvent(this->instIdx);
            return;
        }
        auto intraId = this->devEvtID;
        if constexpr (A_TYPE::ibShare && B_TYPE::ibShare) {
            if (GetSubBlockIdxImpl() == 1) {
                intraId = this->devEvtID - 1;
            }
        }
        WaitEvent(intraId);
    }

    // Only support the mode that the IterateAll is asynchronous and GM output is continuous.
    // In discontinuous scenarios, the system stops responding.
    __aicore__ inline void WaitIterateBatch()
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "WaitIterateBatch not support when enableMixDualMaster is enabled");
        ASSERT(!isSyncGetC); // Must be asynchronous mode
        WaitEvent(this->devEvtID);
    }

    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0,
        bool enSequentialWrite = false, bool waitIterateAll = false, bool fakeMsg = false)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            constexpr uint16_t eventID = 9U;
#if ASCENDC_CPU_DEBUG
            if ASCEND_IS_AIC {
                WaitEvent(eventID);
                cubeObj.cubeObj[0].mul.IterateAll(gm, enAtomic, enSequentialWrite, waitIterateAll, fakeMsg);
                if (sync || waitIterateAll) {
                    NotifyEvent<PIPE_FIX>(cubeObj.cubeObj[0].instID);
                }
                cubeObj.cubeObj[0].mul.End();
                return;
            }
#endif
            NotifyEvent<PIPE_MTE3>(eventID);
            if constexpr(sync) {
                WaitEvent(this->instIdx);
            }
            return;
        }
        TRACE_START(TraceId::KFC_CLIENT_POST_MSG);
        ASSERT(kfcMsg_.body.isFirstIter == 1);
        kfcMsg_.body.iterateFakeMsg = fakeMsg;
        kfcMsg_.body.cAddr = reinterpret_cast<uint64_t>(gm.GetPhyAddr());
        kfcMsg_.body.enAtomic = (uint8_t)(enAtomic);
        kfcMsg_.body.sync = sync;
        kfcMsg_.body.enSequentialWrite = enSequentialWrite;
        kfcMsg_.body.waitIterateAll = waitIterateAll;
        PostMessage<KFC_Enum::MMFUN_ITERATE_ALL, sync>();
        SyncCubeWithVec<A_TYPE::ibShare, B_TYPE::ibShare>();
        if constexpr (sync) {
            auto intraId = this->devEvtID;
            if constexpr (A_TYPE::ibShare && B_TYPE::ibShare) {
                if (GetSubBlockIdxImpl() == 1) {
                    intraId = this->devEvtID - 1;
                }
            }
            WaitEvent(intraId);
        }
        isSyncGetC = sync;
        TRACE_STOP(TraceId::KFC_CLIENT_POST_MSG);
    }

    template <bool sync = true>
    __aicore__ inline void IterateAll(const LocalTensor<DstT>& ubCmatrix, uint8_t enAtomic = 0)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster){
#if (__CCE_AICORE__ == 220)
            ASSERT("IterateAll localTensor not support when enableMixDualMaster is enabled");
#endif
            return;
        }
        TRACE_START(TraceId::KFC_CLIENT_POST_MSG);
        ASSERT(sync == true);
        ASSERT(enAtomic == 0);
        ASSERT(kfcMsg_.body.isFirstIter == 1);
        ASSERT((PhyPosIsL1(C_TYPE::pos)) && "IterateAll LocalTensor only support TPosition A1 or B1");
        ASSERT(!(A_TYPE::ibShare && B_TYPE::ibShare) && "IterateAll LocalTensor not support when sameab"
                                                        " is enabled");
        if (ubCmatrix.GetPosition() == static_cast<int32_t>(TPosition::TSCM)) {
            kfcMsg_.body.cAddr = GetTscmAddr(ubCmatrix);
            kfcMsg_.body.cIsTscm = 1;
        } else {
            kfcMsg_.body.cAddr = GetGlobalAddr<typename C_TYPE::T, false>(ubCmatrix);
        }
        kfcMsg_.body.enAtomic = (uint8_t)(enAtomic);
        kfcMsg_.body.sync = sync;
        ASSERT(kfcMsg_.body.enSequentialWrite == 0);
        GM_ADDR gmDataAddr = reinterpret_cast<GM_ADDR>(kfcMsg_.body.cAddr);
        *((__gm__ uint64_t*)mmCntAddr_) = 0;
        GlobalTensor<uint64_t> mmCntGlobal;
        mmCntGlobal.SetGlobalBuffer((__gm__ uint64_t*)mmCntAddr_);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(mmCntGlobal);
        PostMessage<KFC_Enum::MMFUN_ITERATE_ALL, sync>();

        if constexpr (sync) {
            WaitEvent(this->devEvtID);
            CopyToUB(ubCmatrix, gmDataAddr, ubCmatrix.GetSize());
        }
        isSyncGetC = sync;
        TRACE_STOP(TraceId::KFC_CLIENT_POST_MSG);
    }

    template <bool sync = true, bool waitIterateBatch = false>
    __aicore__ inline void IterateBatch(const GlobalTensor<DstT>& gm, uint32_t batchA, uint32_t batchB,
        bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0,
        const uint32_t matrixStrideC = 0, const bool enPartialSum = false, const uint8_t enAtomic = 0)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "IterateBatch not support when enableMixDualMaster is enabled");
        TRACE_START(TraceId::KFC_CLIENT_POST_MSG);
        ASSERT(kfcMsg_.body.isFirstIter == 1);
        ASSERT(!(A_TYPE::ibShare && B_TYPE::ibShare) && "IterateBatch not support when when sameab"
                                                        " is enabled");
        kfcMsg_.body.cAddr = reinterpret_cast<uint64_t>(gm.GetPhyAddr());
        kfcMsg_.body.enSequentialWrite = enSequentialWrite;
        kfcMsg_.body.sync = sync;
        kfcMsg_.body.batchA = batchA;
        kfcMsg_.body.batchB = batchB;
        kfcMsg_.body.matrixStrideA = matrixStrideA;
        kfcMsg_.body.matrixStrideB = matrixStrideB;
        kfcMsg_.body.matrixStrideC = matrixStrideC;
        kfcMsg_.body.waitIterateBatch = waitIterateBatch;
        kfcMsg_.body.enPartialSum = enPartialSum;
        kfcMsg_.body.enAtomic = (uint8_t)(enAtomic);
        kfcMsg_.body.setBatch = 1;

        *((__gm__ uint64_t*)mmCntAddr_) = 0;
        GlobalTensor<uint64_t> mmCntGlobal;
        mmCntGlobal.SetGlobalBuffer((__gm__ uint64_t*)mmCntAddr_);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(mmCntGlobal);
        PostMessage<KFC_Enum::MMFUN_ITERATE_BATCH_ALL, sync>();

        if constexpr (sync) {
            WaitEvent(this->devEvtID);
        }
        isSyncGetC = sync;
        TRACE_STOP(TraceId::KFC_CLIENT_POST_MSG);
    }

    template <bool sync = true>
    __aicore__ inline void IterateBatch(const LocalTensor<DstT>& ubCmatrix, uint32_t batchA, uint32_t batchB,
        bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0,
        const uint32_t matrixStrideC = 0, const bool enPartialSum = false, const uint8_t enAtomic = 0)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "IterateBatch not support when enableMixDualMaster is enabled");
        TRACE_START(TraceId::KFC_CLIENT_POST_MSG);
        ASSERT(sync == true);
        ASSERT(kfcMsg_.body.isFirstIter == 1);
        ASSERT(!(A_TYPE::ibShare && B_TYPE::ibShare) && "IterateBatch not support when sameab is enabled");
        if (ubCmatrix.GetPosition() == static_cast<int32_t>(TPosition::TSCM)) {
            kfcMsg_.body.cAddr = GetTscmAddr(ubCmatrix);
            kfcMsg_.body.cIsTscm = 1;
        } else {
            kfcMsg_.body.cAddr = GetGlobalAddr<typename C_TYPE::T, false>(ubCmatrix);
        }
        kfcMsg_.body.enSequentialWrite = enSequentialWrite;
        kfcMsg_.body.sync = sync;
        kfcMsg_.body.batchA = batchA;
        kfcMsg_.body.batchB = batchB;
        kfcMsg_.body.matrixStrideA = matrixStrideA;
        kfcMsg_.body.matrixStrideB = matrixStrideB;
        kfcMsg_.body.matrixStrideC = matrixStrideC;
        kfcMsg_.body.enPartialSum = enPartialSum;
        kfcMsg_.body.enAtomic = (uint8_t)(enAtomic);
        kfcMsg_.body.setBatch = 1;
        GM_ADDR gmDataAddr = reinterpret_cast<GM_ADDR>(kfcMsg_.body.cAddr);
        *((__gm__ uint64_t*)mmCntAddr_) = 0;
        GlobalTensor<uint64_t> mmCntGlobal;
        mmCntGlobal.SetGlobalBuffer((__gm__ uint64_t*)mmCntAddr_);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(mmCntGlobal);
        PostMessage<KFC_Enum::MMFUN_ITERATE_BATCH_ALL, sync>();

        if constexpr (sync) {
            WaitEvent(this->devEvtID);
            CopyToUB(ubCmatrix, gmDataAddr, ubCmatrix.GetSize());
        }
        isSyncGetC = sync;
        TRACE_STOP(TraceId::KFC_CLIENT_POST_MSG);
    }

    template <bool sync = true, bool waitIterateBatch = false>
    __aicore__ inline void IterateNBatch(const uint32_t batchLoop, uint32_t batchA, uint32_t batchB,
        bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0,
        const uint32_t matrixStrideC = 0, const bool enPartialSum = false, const uint8_t enAtomic = 0)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "IterateNBatch not support when enableMixDualMaster is enabled");
        if constexpr (!ToMatmulConfig(MM_CFG).isNBatch) {
            return;
        }
        TRACE_START(TraceId::KFC_CLIENT_POST_MSG);
        cntIter_ = 0;
        cOffset_ = 0;
        curProcess = 0;
        ASSERT(kfcMsg_.body.isFirstIter == 1);
        ASSERT(cacheWorkspaceAddr);
        ASSERT(!(A_TYPE::ibShare && B_TYPE::ibShare) && "IterateNBatch not support when sameab is enabled");
        kfcMsg_.body.cAddr = reinterpret_cast<uint64_t>(cacheWorkspaceAddr);
        kfcMsg_.body.enSequentialWrite = enSequentialWrite;
        kfcMsg_.body.sync = sync;
        kfcMsg_.body.batchLoop = batchLoop;
        kfcMsg_.body.batchA = batchA;
        kfcMsg_.body.batchB = batchB;
        kfcMsg_.body.matrixStrideA = matrixStrideA;
        kfcMsg_.body.matrixStrideB = matrixStrideB;
        kfcMsg_.body.matrixStrideC = matrixStrideC;
        kfcMsg_.body.enPartialSum = enPartialSum;
        kfcMsg_.body.enAtomic = (uint8_t)(enAtomic);
        kfcMsg_.body.setBatch = 1;
        kfcMsg_.body.waitIterateBatch = waitIterateBatch;
        *((__gm__ uint64_t*)mmCntAddr_) = 0;
        GlobalTensor<uint64_t> mmCntGlobal;
        mmCntGlobal.SetGlobalBuffer((__gm__ uint64_t*)mmCntAddr_);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(mmCntGlobal);
        PostMessage<KFC_Enum::MMFUN_ITERATE_N_BATCH_ALL, sync>();
        if constexpr (sync) {
            WaitEvent(this->devEvtID);
        }
        isSyncGetC = sync;
        TRACE_STOP(TraceId::KFC_CLIENT_POST_MSG);
    }

    // Synchronous interface. The user sends the GM address, which contains 64 bits.
    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0,
        bool enSequentialWrite = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "GetTensorC not support when enableMixDualMaster is enabled");
        TRACE_START(TraceId::KFC_CLIENT_REV_MSG_GM);
        ASSERT(kfcMsg_.body.isFirstIter == 0);
        ASSERT(isSyncGetC); // The mode must be synchronous.

        kfcMsg_.body.cAddr = reinterpret_cast<uint64_t>(gm.GetPhyAddr());
        kfcMsg_.body.enAtomic = (uint8_t)(enAtomic);
        kfcMsg_.body.enSequentialWrite = enSequentialWrite;
        kfcMsg_.body.sync = sync;

        PostMessage<KFC_Enum::MMFUN_GET_TENSOR_C, sync>();

        if constexpr (sync) {
            WaitEvent(this->devEvtID);
        }
        TRACE_STOP(TraceId::KFC_CLIENT_REV_MSG_GM);
    }
    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& co2Local,
        uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "GetTensorC not support when enableMixDualMaster is enabled");
        TRACE_START(TraceId::KFC_CLIENT_REV_MSG_GM);
        ASSERT(kfcMsg_.body.isFirstIter == 0);
        ASSERT(isSyncGetC); // must synchronization mode

        kfcMsg_.body.cAddr = reinterpret_cast<uint64_t>(gm.GetPhyAddr());
        kfcMsg_.body.enAtomic = (uint8_t)enAtomic;
        kfcMsg_.body.enSequentialWrite = enSequentialWrite;
        kfcMsg_.body.sync = sync;

        PostMessage<KFC_Enum::MMFUN_GET_TENSOR_C, sync>();

        if constexpr (sync) {
            WaitEvent(this->devEvtID);
        }

        CopyToUB(co2Local, gm.GetPhyAddr(), co2Local.GetSize());
        TRACE_STOP(TraceId::KFC_CLIENT_REV_MSG_GM);
    }

    // Synchronous interface
    template <bool sync = true, bool doPad = false>
    __aicore__ inline void GetTensorC(const LocalTensor<DstT>& c, uint8_t enAtomic = 0,
        bool enSequentialWrite = false, uint32_t height = 0, uint32_t width = 0, uint32_t srcGap = 0,
        uint32_t dstGap = 0)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "GetTensorC not support when enableMixDualMaster is enabled");
        TRACE_START(TraceId::KFC_CLIENT_REV_MSG_UB);
        ASSERT(kfcMsg_.body.isFirstIter == 0);
        if (!isSyncGetC) { // Asynchronous
            ASSERT(cacheWorkspaceAddr);
            ASSERT(enAtomic == 0);

            if (curProcess < INC_PROCESS_CHECK) {
                ++curProcess;
                WaitEvent(this->devEvtID);
            }

            uint32_t size;
            if constexpr (ToMatmulConfig(MM_CFG).baseMN != 0) {
                size = ToMatmulConfig(MM_CFG).baseMN * sizeof(typename C_TYPE::T);
            } else {
                size = cubeTiling.GetBaseM() * cubeTiling.GetBaseN() * sizeof(typename C_TYPE::T);
            }
            if constexpr (doPad) {
                CopyToUBPad(c, cacheWorkspaceAddr + cOffset_, height, width, srcGap, dstGap);
            } else {
                CopyToUB(c, cacheWorkspaceAddr + cOffset_, c.GetSize());
            }
            cOffset_ += size;
            TRACE_STOP(TraceId::KFC_CLIENT_REV_MSG_UB);
            return;
        }

        ASSERT(sync == true); // must be the same as Iterate.
        ASSERT(enAtomic == 0);
        kfcMsg_.body.cAddr = GetGlobalAddr<typename C_TYPE::T, false>(c);
        kfcMsg_.body.sync = 1;
        kfcMsg_.body.enAtomic = (uint8_t)(enAtomic);
        kfcMsg_.body.enSequentialWrite = enSequentialWrite;

        GM_ADDR gmDataAddr = reinterpret_cast<GM_ADDR>(kfcMsg_.body.cAddr);
        PostMessage<KFC_Enum::MMFUN_GET_TENSOR_C, true>();

        WaitEvent(this->devEvtID);

        if constexpr (PhyPosIsUB(C_TYPE::pos)) {
            if constexpr (doPad) {
                CopyToUBPad(c, (__gm__ DstT*)gmDataAddr, height, width);
            } else {
                CopyToUB(c, (__gm__ DstT*)gmDataAddr, c.GetSize());
            }
        }
        TRACE_STOP(TraceId::KFC_CLIENT_REV_MSG_UB);
        return;
    }

    template <bool sync = true>
    __aicore__ inline GlobalTensor<DstT> GetTensorC(uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "GetTensorC not support when enableMixDualMaster is enabled");
        TRACE_START(TraceId::KFC_CLIENT_REV_MSG_GM);
        ASSERT(kfcMsg_.body.isFirstIter == 0);
        ASSERT(!isSyncGetC); // Asynchronous only
        ASSERT(cacheWorkspaceAddr);
        if (curProcess < INC_PROCESS_CHECK) {
            ++curProcess;
            auto intraId = this->devEvtID;
            if constexpr (A_TYPE::ibShare && B_TYPE::ibShare) {
                if (GetSubBlockIdxImpl() == 1) {
                    intraId = this->devEvtID - 1;
                }
            }
            WaitEvent(intraId);
        }
        uint32_t size;
        GlobalTensor<DstT> global;
        if constexpr (ToMatmulConfig(MM_CFG).baseMN != 0) {
            size = ToMatmulConfig(MM_CFG).baseMN * sizeof(typename C_TYPE::T);
            global.SetGlobalBuffer(reinterpret_cast<__gm__ DstT *>(cacheWorkspaceAddr + cOffset_),
                ToMatmulConfig(MM_CFG).baseMN);
        } else {
            size = cubeTiling.GetBaseM() * cubeTiling.GetBaseN() * sizeof(typename C_TYPE::T);
            global.SetGlobalBuffer(reinterpret_cast<__gm__ DstT *>(cacheWorkspaceAddr + cOffset_),
                cubeTiling.GetBaseM() * cubeTiling.GetBaseN());
        }
        cOffset_ += size;
        TRACE_STOP(TraceId::KFC_CLIENT_REV_MSG_GM);
        return global;
    }

    template <bool sync = true>
    __aicore__ inline GlobalTensor<DstT> GetBatchTensorC(uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "GetBatchTensorC not support when enableMixDualMaster is enabled");
        GlobalTensor<DstT> global;
        if constexpr (!ToMatmulConfig(MM_CFG).isNBatch) {
            return global;
        }
        TRACE_START(TraceId::KFC_CLIENT_REV_MSG_GM);
        ASSERT(kfcMsg_.body.isFirstIter == 0);
        ASSERT(!isSyncGetC); // only support async
        ASSERT(cacheWorkspaceAddr);
        if (curProcess < INC_PROCESS_CHECK) {
            ++curProcess;
            WaitEvent(this->devEvtID);
        }

        uint32_t batch = batchA > batchB ? batchA : batchB;
        uint32_t size = batch * cubeTiling.GetSingleCoreM() * cubeTiling.GetSingleCoreN() * sizeof(typename C_TYPE::T);
        global.SetGlobalBuffer(reinterpret_cast<__gm__ DstT *>(cacheWorkspaceAddr + cOffset_),
            batch * cubeTiling.GetSingleCoreM() * cubeTiling.GetSingleCoreN());
        cOffset_ += size;
        TRACE_STOP(TraceId::KFC_CLIENT_REV_MSG_GM);
        return global;
    }

    template <bool sync = true>
    __aicore__ inline GlobalTensor<DstT> GetBatchC(uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "GetBatchC not support when enableMixDualMaster is enabled");
        return GetBatchTensorC(batchA, batchB, enSequentialWrite);
    }

    // coordinated use with IterateNBatch, get single IterateBatch outcome
    template <bool sync = true>
    __aicore__ inline void GetBatchTensorC(const LocalTensor<DstT>& c, uint32_t batchA, uint32_t batchB,
        bool enSequentialWrite = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "GetBatchTensorC not support when enableMixDualMaster is enabled");
        if constexpr (!ToMatmulConfig(MM_CFG).isNBatch) {
            return;
        }
        TRACE_START(TraceId::KFC_CLIENT_REV_MSG_GM);
        ASSERT(kfcMsg_.body.isFirstIter == 0);
        ASSERT(cacheWorkspaceAddr);
        ASSERT(enSequentialWrite);
        ASSERT(!isSyncGetC); // only support async

        if (curProcess < INC_PROCESS_CHECK) {
            ++curProcess;
            WaitEvent(this->devEvtID);
        }

        uint32_t batch = batchA > batchB ? batchA : batchB;
        uint32_t size = batch * cubeTiling.GetSingleCoreM() * cubeTiling.GetSingleCoreN() * sizeof(typename C_TYPE::T);
        CopyToUB(c, cacheWorkspaceAddr + cOffset_, c.GetSize());
        cOffset_ += size;
        TRACE_STOP(TraceId::KFC_CLIENT_REV_MSG_GM);
    }

    template <bool sync = true>
    __aicore__ inline void GetBatchC(const LocalTensor<DstT>& c, uint32_t batchA, uint32_t batchB,
        bool enSequentialWrite = false)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "GetBatchC not support when enableMixDualMaster is enabled");
        GetBatchTensorC(c, batchA, batchB, enSequentialWrite);
    }

    __aicore__ inline void AsyncGetTensorC(const LocalTensor<DstT>& c)
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "AsyncGetTensorC not support when enableMixDualMaster is enabled");
        TRACE_START(TraceId::KFC_CLIENT_REV_MSG_GM);
        ASSERT(kfcMsg_.body.isFirstIter == 0);
        ASSERT(!isSyncGetC);
        ASSERT(cacheWorkspaceAddr);

        if (curProcess < INC_PROCESS_CHECK) {
            ++curProcess;
            WaitEvent(this->devEvtID);
        }

        uint32_t size = cubeTiling.GetBaseM() * cubeTiling.GetBaseN() * sizeof(typename C_TYPE::T);
        CopyToUB<DstT, uint8_t, false>(c, cacheWorkspaceAddr + cOffset_, c.GetSize());
        cOffset_ += size;
        TRACE_STOP(TraceId::KFC_CLIENT_REV_MSG_GM);
        return;
    }

    __aicore__ inline void WaitGetTensorC()
    {
        ASSERT(!ToMatmulConfig(MM_CFG).enableMixDualMaster && 
            "WaitGetTensorC not support when enableMixDualMaster is enabled");
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
    }

    template <bool isTurnOnDebug = true>
    __aicore__ inline MatrixOffset GetOffsetC()
    {
        if constexpr (isTurnOnDebug) {
            static_assert(!isTurnOnDebug, "unsupported!");
        }
    }

    __aicore__ inline void SetLocalWorkspace(const LocalTensor<uint8_t>& tmpBuffer) {};
#if ASCENDC_CPU_DEBUG
public:
    // this is useless code just for cpu debug
    typename MatmulInstAux<IsSharedObj(MM_CFG),
                                   A_TYPE,
                                   B_TYPE,
                                   C_TYPE,
                                   BIAS_TYPE,
                                   MM_CFG,
                                   MM_CB,
                                   MATMUL_POLICY>::MATMUL cubeObj;
#endif

private:
    __gm__ KfcMsg* mmCntAddr_;
    GM_ADDR cacheWorkspaceAddr;
    // Multiple instances with only one message queue maintained.
    // Use shared memory to get the queue.
    KfcCommClient* client;
    TPipe* tpipe;
    MatmulTiling<MM_CFG> cubeTiling;
    KfcMsg kfcMsg_;

    bool isSyncGetC;
    uint16_t devEvtID;
    uint16_t instIdx;
    uint16_t curProcess;

    uint32_t mIter_;
    uint32_t nIter_;
    uint32_t cntIter_;
    uint32_t mnIter_;
    uint64_t cOffset_;
    template <class T, class U>
    friend __aicore__ inline void InitKfcClient(T& cubeObj, U *tiling, TPipe *tpipe, KfcCommClient *client, int instIdx,
        GM_ADDR workspace);
    template <class... Args> friend struct AscendC::GetCubeObjConfig;
    constexpr static bool enableMixDualMaster = ToMatmulConfig(MM_CFG).enableMixDualMaster;
    constexpr static bool enableABShare = A_TYPE::ibShare && B_TYPE::ibShare;
private:
    __aicore__ inline void InitStatic()
    {
        if (ToMatmulConfig(MM_CFG).singleCoreM == 0 && this->cubeTiling.IsNull()) {
            return;
        }
        ASSERT(sizeof(KfcMsg) % CACHE_LINE_SIZE == 0);

        *((uint64_t*)&kfcMsg_) = 0;
        *((uint64_t*)&(kfcMsg_.body)) = 0;
        nIter_ = ConstCeil(this->cubeTiling.GetSingleCoreN(), this->cubeTiling.GetBaseN());
        mIter_ = ConstCeil(this->cubeTiling.GetSingleCoreM(), this->cubeTiling.GetBaseM());
        mnIter_ = nIter_ * mIter_;
        cacheWorkspaceAddr = nullptr;
    }

    template <class T> __aicore__ inline uint64_t CopyGlobalAddr(GM_ADDR& gmDataAddr, const LocalTensor<T>& data)
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);

        struct DataCopyParams param;
        param.blockLen = data.GetSize() / AscendCUtils::GetC0Count(sizeof(T));
        GlobalTensor<T> globalTensor;
        globalTensor.SetGlobalBuffer((__gm__ T*)gmDataAddr);
        DataCopy(globalTensor, data, param);

        return reinterpret_cast<uint64_t>(gmDataAddr);
    }

    template <class T, bool isCopy> __aicore__ inline uint64_t GetGlobalAddr(
        const LocalTensor<T>& data)
    {
        uint64_t size = Ceil(data.GetSize() * sizeof(T), ONE_BLK_SIZE) * ONE_BLK_SIZE;
        if constexpr (IsSameType<T, int4b_t>::value) {
            size /= INT4_TWO;
        }
        auto gmDataAddr = client->AllocUB(size, kfcMsg_.ubAddr);

        if constexpr (isCopy) {
            return CopyGlobalAddr(gmDataAddr, data);
        }
        return reinterpret_cast<uint64_t>(gmDataAddr);
    }
    template <class T> __aicore__ inline uint64_t GetTscmAddr(const LocalTensor<T>& data)
    {
#if ASCENDC_CPU_DEBUG
        ASSERT(GetTPipePtr() != nullptr && "tpipe cannot be nullptr when matmul client post msg");
        return GetAbsAddr<T>(GetTPipePtr(), data);
#else
        return (uint64_t)data.GetPhyAddr();
#endif
    }
    template <KFC_Enum funID, bool isAck> __aicore__ inline void PostMessage()
    {
        if constexpr (A_TYPE::ibShare && B_TYPE::ibShare) {
            ASSERT(DoMatmulNorm(MM_CFG) && "MM_CFG should use norm config when sameab is enabled");
            if (GetSubBlockIdxImpl() == 1) { // Do not send v1's message to cube
                *((uint32_t *)&kfcMsg_.body) = 0; // Clear all flag bits.
                kfcMsg_.ubAddr = -1;
                return;
            }
        }
        kfcMsg_.head = KfcMsgMakeFlag(funID, this->instIdx);

        auto msg = client->AllocMessage();
        ASSERT(msg != nullptr && "msg cannot be nullptr when matmul client post msg");

        auto tmp1 = reinterpret_cast<__ubuf__ uint64_t*>(client->ubMsg);
        auto tmp2 = reinterpret_cast<uint64_t*>(&kfcMsg_);
        for (int i = 0; i < sizeof(kfcMsg_) / sizeof(uint64_t); i++, tmp1++, tmp2++) {
            *tmp1 = *tmp2;
        }

        client->PostMessage<isAck>(msg);

        // clear flag
        *((uint32_t*)&kfcMsg_.body) = 0;  // Clear all flag bits.
        kfcMsg_.ubAddr = -1;
    }

    // height width in unit of element
    template <class T, class U, bool sync = true>
    __aicore__ inline void CopyToUBPad(const LocalTensor<T>& data, const __gm__ U* addr, uint32_t height = 0,
        uint32_t width = 0, uint32_t srcGap = 0, uint32_t dstGap = 0)
    {
        ASSERT(C_TYPE::format == CubeFormat::ND_ALIGN &&
            "Only support padding in ND_ALIGN mode, please check template param of GetTensorC.");

        DataCopyParams copyParams{ static_cast<uint16_t>(height), static_cast<uint16_t>(width * sizeof(T)),
            static_cast<uint16_t>(srcGap), static_cast<uint16_t>(dstGap) };
        DataCopyPadParams padParams{ true, 0,
            static_cast<uint8_t>(
            ConstCeil(width, AscendCUtils::GetC0Count(sizeof(T))) * AscendCUtils::GetC0Count(sizeof(T)) - width),
            0 };
        GlobalTensor<T> globalTensor;
        globalTensor.SetGlobalBuffer((__gm__ T*)addr);
        DataCopyPad(data, globalTensor, copyParams, padParams);

        if constexpr (sync) {
            event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventID);
            WaitFlag<HardEvent::MTE2_V>(eventID);
        }
    }

    template <class T, class U, bool sync = true>
    __aicore__ inline void CopyToUB(const LocalTensor<T>& data, const __gm__ U* addr, uint32_t size)
    {
        struct DataCopyParams repeatParams;
        repeatParams.blockLen = size / AscendCUtils::GetC0Count(sizeof(T));
        GlobalTensor<T> globalTensor;
        globalTensor.SetGlobalBuffer((__gm__ T*)addr);
        if constexpr (C_TYPE::format == CubeFormat::ND_ALIGN) {
            int32_t batchNum = 1;
            int32_t offset = 0;
            if constexpr (C_TYPE::layout != LayoutMode::NONE) {
                int32_t alignedSingleCoreN = ConstCeil(cubeTiling.GetSingleCoreN(), AscendCUtils::GetC0Count(sizeof(T))) *
                    AscendCUtils::GetC0Count(sizeof(T));
                offset = cubeTiling.GetSingleCoreM()  * alignedSingleCoreN;
                batchNum = size / offset;
            }
            for (int32_t idx = 0; idx < batchNum; ++idx) {
                DataCopyParams copyParams{ static_cast<uint16_t>(cubeTiling.GetSingleCoreM()),
                    static_cast<uint16_t>(cubeTiling.GetSingleCoreN() * sizeof(T)), 0, 0 };
                DataCopyPadParams padParams{ true, 0,
                    static_cast<uint8_t>(ConstCeil(cubeTiling.GetSingleCoreN(), AscendCUtils::GetC0Count(sizeof(T))) *
                    AscendCUtils::GetC0Count(sizeof(T)) -
                    cubeTiling.GetSingleCoreN()),
                    0 };
                DataCopyPad(data[idx * offset], globalTensor[idx * offset], copyParams, padParams);
            }
        } else {
            DataCopy(data, globalTensor, repeatParams);
        }

        if constexpr (sync) {
            event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventID);
            WaitFlag<HardEvent::MTE2_V>(eventID);
        }
    }

    template <class T>
    __aicore__ inline uint64_t GetGMAddrAndCopyUB(const __gm__ T* gmDataAddr, const LocalTensor<T>& data)
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);

        struct DataCopyParams param;
        param.blockLen = data.GetSize() / AscendCUtils::GetC0Count(sizeof(T));
        GlobalTensor<T> globalTensor;
        globalTensor.SetGlobalBuffer((__gm__ T*)gmDataAddr);
        DataCopy(globalTensor, data, param);

        return reinterpret_cast<uint64_t>(gmDataAddr);
    }
};

// Match Policy with CallBack paramter
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
class MatmulClient
: public MatmulClientBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY> {
public:
    __aicore__ inline MatmulClient() {}
};
} // namespace matmul
#endif
