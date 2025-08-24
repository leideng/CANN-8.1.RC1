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
 * \file matmul_server.h
 * \brief
 */
#ifndef IMPL_MATMUL_KFC_MATMUL_SERVER_H
#define IMPL_MATMUL_KFC_MATMUL_SERVER_H

#include "matmul_server_utils.h"

namespace AscendC {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG = CFG_NORM,
    class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulService {
    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    using SrcT = typename A_TYPE::T;
    using DstT = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

public:
    __aicore__ inline MatmulService() {}
    __aicore__ inline void InitKfc(TPipe* tpipe, void* tiling, KfcCommServer* kfc, int32_t instID, GM_ADDR workspace)
    {
        ASSERT(instID >= 0 && "instID should be not less than 0 when init kfc matmul server");
        this->instID = instID;
        if constexpr (!ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            ASSERT(kfc != nullptr && "kfc cannot be nullptr when init kfc matmul server");
            ASSERT(workspace != nullptr && "workspace cannot be nullptr when init kfc matmul server");
            this->kfcCommSrv = kfc;
            this->workspace = workspace;
            mul.SetSubBlockIdx(kfcCommSrv->subBlockID);
            if constexpr (!ToMatmulConfig(MM_CFG).enableInit) {
                msgAux.msg0.setOrgShape = false;
                msgAux.msg1.setOrgShape = false;
            }
            this->devEvtID = instID;
            if constexpr (A_TYPE::ibShare == true || B_TYPE::ibShare == true) {
                if (kfcCommSrv->subBlockID == 0) {
                    gCache.Init();
                }
            }
        } else {
            mul.SetSubBlockIdx(0);
        }
        using TILING_TYPE = typename std::remove_cv<typename std::remove_reference<decltype(MM_CFG)>::type>::type;
        if constexpr (IsSameTypeV<TILING_TYPE, MatmulApiStaticTiling>) {
            tiling_.SetTiling((TCubeTiling *)tiling);
            mul.Init(tiling_.GetTiling(), nullptr);
        } else if (tiling) {
            tiling_.SetTiling((TCubeTiling *)tiling);
            mul.Init(tiling_.GetTiling(), nullptr);
        }
    }

    __aicore__ inline void Init(__gm__ KfcMsg* msg);
    __aicore__ inline void SetSubBlockIdx(uint8_t idx)
    {
        mul.SetSubBlockIdx(idx);
    }

    __aicore__ inline void SetOrgShape(__gm__ KfcMsg* msg);
    __aicore__ inline void SetSingleShape(__gm__ KfcMsg* msg)
    {
        if (msg->body.setTail) {
            mul.SetSingleShape(msg->body.singleM, msg->body.singleN, msg->body.singleK);
        }
    }

    __aicore__ inline void SetTail(__gm__ KfcMsg* msg)
    {
        if (msg->body.setTail) {
            mul.SetTail(msg->body.singleM, msg->body.singleN, msg->body.singleK);
        }
    }

    __aicore__ inline void SetHF32(__gm__ KfcMsg* msg)
    {
        mul.SetHF32(static_cast<bool>(msg->body.enHF32), static_cast<int32_t>(msg->body.hf32TransMode));
    }

    __aicore__ inline void SetTensorA(__gm__ KfcMsg* msg);
    __aicore__ inline void SetTensorA(__gm__ KfcMsg* msg, const uint64_t size, const uint64_t offset);
    __aicore__ inline void SetQuantVector(__gm__ KfcMsg* msg)
    {
        if (!msg->body.setQuant) {
            return;
        }
        int quantMode = msg->body.quantMode;
        if (quantMode == 1) {
            uint64_t quantScalar = msg->body.quantScalar;
            mul.SetQuantScalar(quantScalar);
        } else if (quantMode == 2) {
            const uint64_t size = static_cast<uint64_t>(msg->body.quantSize);
            GlobalTensor<uint64_t> quantGlobal;
            quantGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(msg->body.quantAddr), size);
            mul.SetQuantVector(quantGlobal);
        }
    }

    __aicore__ inline void SetBatchNum(__gm__ KfcMsg* msg)
    {
        if constexpr (A_TYPE::layout == LayoutMode::NONE) {
            return;
        }
        if (!msg->body.setBatch) {
            return;
        }
        mul.SetBatchNum(msg->body.batchA, msg->body.batchB);
    }

    __aicore__ inline void SetSelfDefineData(__gm__ KfcMsg* msg)
    {
        GlobalTensor<int64_t> msgGlobal;
        msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(msg) + sizeof(int64_t));
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(msgGlobal);
        mul.SetSelfDefineData(msg->body.dataPtr);
        if constexpr (!ToMatmulConfig(MM_CFG).enableReuse) {
            GlobalTensor<uint32_t> dataGlobal;
            dataGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(msg->body.dataPtr));
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(dataGlobal);
        }
    }

    __aicore__ inline void SetUserDefInfo(__gm__ KfcMsg* msg)
    {
        mul.SetUserDefInfo(msg->userDefInfo.tilingPtr);
    }

    __aicore__ inline void SetTensorB(__gm__ KfcMsg* msg);
    __aicore__ inline void SetTensorB(__gm__ KfcMsg* msg, const uint64_t size, const uint64_t offset);
    __aicore__ inline void SetBias(__gm__ KfcMsg* msg);
    __aicore__ inline void SetBias(__gm__ KfcMsg* msg, const uint64_t offset);
    __aicore__ inline bool GetTensorC(__gm__ KfcMsg* msg);
    __aicore__ inline uint16_t GetInstID()
    {
        return instID;
    }
    __aicore__ inline void IterateSetMessage(__gm__ KfcMsg* msg)
    {
        if constexpr (!ToMatmulConfig(MM_CFG).enableInit) {
            if (mul.GetSubBlockIdx() == 0 && msgAux.msg0.setOrgShape) {
                mul.SetOrgShape(msgAux.msg0.orgM, msgAux.msg0.orgN, msgAux.msg0.orgKa,
                    msgAux.msg0.orgKb, msgAux.msg0.orgKc);
            } else if (mul.GetSubBlockIdx() == 1 && msgAux.msg1.setOrgShape) {
                mul.SetOrgShape(msgAux.msg1.orgM, msgAux.msg1.orgN, msgAux.msg1.orgKa,
                    msgAux.msg1.orgKb, msgAux.msg1.orgKc);
            }
        }
        if (msg->body.isFirstIter) {
            SetTensorA(msg);
            SetTensorB(msg);
            if constexpr (ToMatmulConfig(MM_CFG).enableSetBias) {
                SetBias(msg);
            }
            if constexpr (ToMatmulConfig(MM_CFG).enableSetTail) {
                SetTail(msg);
            }
            if constexpr (ToMatmulConfig(MM_CFG).enableQuantVector) {
                SetQuantVector(msg);
            }
            if constexpr (((ToMatmulConfig(MM_CFG).iterateMode & IterateMode::ITERATE_MODE_BATCH) != 0) ||
                ((ToMatmulConfig(MM_CFG).iterateMode & IterateMode::ITERATE_MODE_N_BATCH) != 0)) {
                if constexpr (A_TYPE::layout != LayoutMode::NONE) {
                    SetBatchNum(msg);
                }
            }
            if constexpr (ToMatmulConfig(MM_CFG).enableSetDefineData) {
                SetSelfDefineData(msg);
            }
        }
    }

    __aicore__ inline void IterateSetMessage(__gm__ KfcMsg* msg, const uint64_t batchASize, const uint64_t batchBSize,
        const uint64_t offsetA = 0, const uint64_t offsetB = 0, const uint64_t offsetBias = 0)
    {
        if (msg->body.isFirstIter) {
            SetTensorA(msg, batchASize, offsetA);
            SetTensorB(msg, batchBSize, offsetB);
            SetBias(msg, offsetBias);
            SetTail(msg);
            SetQuantVector(msg);
            if constexpr (A_TYPE::layout != LayoutMode::NONE) {
                SetBatchNum(msg);
            }
        }
    }

    __aicore__ inline bool IterateBatch(__gm__ KfcMsg* msg);
    __aicore__ inline void StartIterateNBatch(__gm__ KfcMsg* msg, uint32_t &cntIterator);
    __aicore__ inline bool IterateNBatch(__gm__ KfcMsg* msg);
    __aicore__ inline void GetOffsetSize(__gm__ KfcMsg* msg, KFC_Enum funID, uint32_t sync,
        uint64_t &offsetSize, uint32_t &enSequentialWrite);
    __aicore__ inline bool StartIterate(__gm__ KfcMsg* msg, KFC_Enum funID, uint32_t sync, uint32_t &cntIterator);
    __aicore__ inline bool Iterate(__gm__ KfcMsg* msg, KFC_Enum funID);
    __aicore__ inline void QuantCacheRefresh(__gm__ KfcMsg* msg)
    {
        if constexpr (((IsSameType<SrcT, int4b_t>::value || IsSameType<SrcT, int8_t>::value) &&
            IsSameType<DstT, half>::value) ||
            ((IsSameType<SrcT, half>::value || IsSameType<SrcT, bfloat16_t>::value) &&
            IsSameType<DstT, int8_t>::value) ||
            (IsSameType<SrcT, int8_t>::value && (IsSameType<DstT, uint8_t>::value ||
            IsSameType<DstT, int8_t>::value))) {
            GlobalTensor<int64_t> msgGlobal;
            msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(msg) + sizeof(int64_t));
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(msgGlobal);
        }
    }

    __aicore__ inline bool IterateIntraBlockPartSum(__gm__ KfcMsg* msg, KFC_Enum funID)
    {
        if constexpr (A_TYPE::layout != LayoutMode::NONE) {
            return true;
        }
        if constexpr (((IsSameType<SrcT, int8_t>::value || IsSameType<SrcT, int4b_t>::value) &&
           IsSameType<DstT, half>::value) ||
           ((IsSameType<SrcT, half>::value || IsSameType<SrcT, bfloat16_t>::value) &&
           IsSameType<DstT, int8_t>::value) ||
           (IsSameType<SrcT, int8_t>::value && (IsSameType<DstT, int8_t>::value ||
           IsSameType<DstT, uint8_t>::value))) {
            GlobalTensor<int64_t> msgGlobal;
            msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(msg) + sizeof(int64_t));
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(msgGlobal);
        }
        IterateSetMessage(msg);
        if (mul.GetSubBlockIdx() == 0) {
            return true;
        }
        uint64_t size;
        if constexpr (ToMatmulConfig(MM_CFG).singleCoreMN != 0) {
            size = ToMatmulConfig(MM_CFG).singleCoreMN;
        } else {
            size = tiling_.GetSingleCoreM() * tiling_.GetSingleCoreN();
        }

        GlobalTensor<DstT> cGlobal;
        cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(msg->body.cAddr), size);
        mul.IterateAll(cGlobal, msg->body.enAtomic, msg->body.enSequentialWrite,
            msg->body.waitIterateAll, msg->body.iterateFakeMsg);

        uint16_t eventID0 = static_cast<uint16_t>(this->devEvtID * 2 + 0);
        uint16_t eventID1 = static_cast<uint16_t>(this->devEvtID * 2 + 1);
        if (msg->body.sync || msg->body.waitIterateAll) {
            ASSERT(funID == KFC_Enum::MMFUN_ITERATE_ALL);
            NotifyEvent<PIPE_FIX>(eventID0);
            NotifyEvent<PIPE_FIX>(eventID1);
        }
        if (!msg->body.iterateFakeMsg) {
            mul.End();
        }
        TRACE_STOP(TraceId::MatMul_CALC);
        return true;
    }

    __aicore__ inline bool IsSharedObj()
    {
        if constexpr (!ToMatmulConfig(MM_CFG).enableInit || ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            return true;
        }
        return false;
    }

    __aicore__ inline bool IsEnableMixHdAbility()
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster) {
            return true;
        }
        return false;
    }

    template <uint8_t enableHardPoll = 0>
    __aicore__ inline bool SkipMsg(KFC_Enum funID, bool &freeMsg, int &lastMsgId, const int subBlockID)
    {
        if constexpr (A_TYPE::ibShare && B_TYPE::ibShare) {
            return false;
        }
        if constexpr (A_TYPE::ibShare || B_TYPE::ibShare || ToMatmulConfig(MM_CFG).intraBlockPartSum) {
            if (funID == KFC_Enum::MMFUN_ITERATE_ALL) {
                if (lastMsgId == subBlockID) {
                    freeMsg = false;
                    return true;
                }
                lastMsgId = subBlockID;
                return false;
            }
            return false;
        } else {
            return false;
        }
    }

    template <uint8_t enableHardPoll = 0>
    __aicore__ inline bool LockMsgQueue(KFC_Enum funID, bool &freeMsg, int &lastMsgId, const int subBlockID,
        __gm__ KfcMsg *msg = nullptr)
    {
        if constexpr (A_TYPE::ibShare && B_TYPE::ibShare) {
            return true;
        }
        return false;
    }

    __aicore__ inline bool Process(__gm__ KfcMsg* msg, KFC_Enum funID)
    {
        if constexpr (((ToMatmulConfig(MM_CFG).iterateMode & IterateMode::ITERATE_MODE_ALL) != 0) ||
            ((ToMatmulConfig(MM_CFG).iterateMode & IterateMode::ITERATE_MODE_NORMAL) != 0)) {
            if ((static_cast<uint16_t>(funID) & static_cast<uint16_t>(KFC_Enum::MMFUN_MASK)) ==
                static_cast<uint16_t>(KFC_Enum::MMFUN_MASK)) {
                if constexpr (ToMatmulConfig(MM_CFG).intraBlockPartSum) {
                    return IterateIntraBlockPartSum(msg, funID);
                } else {
                    return Iterate(msg, funID);
                }
            }
        }
        if constexpr (((ToMatmulConfig(MM_CFG).iterateMode & IterateMode::ITERATE_MODE_BATCH) != 0) &&
                    (A_TYPE::layout != LayoutMode::NONE)) {
            if (funID == KFC_Enum::MMFUN_ITERATE_BATCH_ALL) {
                return IterateBatch(msg);
            }
        }
        if constexpr (ToMatmulConfig(MM_CFG).enableEnd) {
            if (funID == KFC_Enum::MMFUN_END) {
                mul.End();
            }
        }
        if constexpr (ToMatmulConfig(MM_CFG).enableGetTensorC) {
            if (funID == KFC_Enum::MMFUN_GET_TENSOR_C) {
                return GetTensorC(msg);
            }
        }
        if constexpr (ToMatmulConfig(MM_CFG).enableSetOrgShape) {
            if (funID == KFC_Enum::MMFUN_SET_ORG_SHAPE) {
                SetOrgShape(msg);
                return true;
            }
        }
        if constexpr (ToMatmulConfig(MM_CFG).enableInit) {
            if (funID == KFC_Enum::MMFUN_INIT) {
                Init(msg);
                return true;
            }
        }
        if constexpr (((ToMatmulConfig(MM_CFG).iterateMode & IterateMode::ITERATE_MODE_N_BATCH) != 0) &&
                      (A_TYPE::layout != LayoutMode::NONE)) {
            if (funID == KFC_Enum::MMFUN_ITERATE_N_BATCH_ALL) {
                return IterateNBatch(msg);
            }
        }
        if (funID == KFC_Enum::MMFUN_SET_USER_DEF_INFO) {
            SetUserDefInfo(msg);
            return true;
        }
        if (funID == KFC_Enum::MMFUN_SET_HF32) {
            SetHF32(msg);
            return true;
        }
        ASSERT("illegal function ID.");
        return true;
    }

    template <class T> __aicore__ LocalTensor<T> GetTscmTensor(uint64_t addr, const uint64_t size)
    {
        LocalTensor<T> scmLocal;
        TBuffAddr scmTbuf;
        scmTbuf.logicPos = (uint8_t)(TPosition::TSCM);
        scmTbuf.dataLen = size * sizeof(DstT);
        scmTbuf.bufferAddr = addr;
#if ASCENDC_CPU_DEBUG
        scmTbuf.absAddr = GetTPipePtr()->GetBaseAddr((uint8_t)(TPosition::TSCM)) + addr;
#endif
        scmLocal.SetAddr(scmTbuf);
        return scmLocal;
    }

public:
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY> mul;
private:
    GM_ADDR workspace;
    KfcCommServer* kfcCommSrv;
    MatmulTiling<MM_CFG> tiling_;
    TCubeTiling tmpTiling_; // for compatible with init interface
    typename IBShareCache<IsIBShare<A_TYPE, B_TYPE>()>::ShareCache gCache;
    typename ShareMatmulAux<!ToMatmulConfig(MM_CFG).enableInit>::MSG msgAux;
public:
    uint16_t instID;
private:
    uint16_t devEvtID;
};
} // namespace AscendC
#endif // __MATMUL_SERVER_H__