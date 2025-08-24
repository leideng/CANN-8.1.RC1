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
 * \file matmul_server_impl.h
 * \brief
 */
#ifndef IMPL_MATMUL_KFC_MATMUL_SERVER_IMPL_H
#define IMPL_MATMUL_KFC_MATMUL_SERVER_IMPL_H

#include "matmul_server.h"

namespace AscendC {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::Init(__gm__ KfcMsg* msg)
{
    if constexpr (!ToMatmulConfig(MM_CFG).enableInit) {
        return;
    } else {
        ASSERT(msg != nullptr && "msg cannot be nullptr when init matmul server");
        ASSERT(msg->tilingInfo.tilingAddr != nullptr && "tiling cannot be nullptr when init matmul server");
        auto temp1 = ((__gm__ uint32_t*)(msg->tilingInfo.tilingAddr));
        tiling_.SetTiling(&tmpTiling_);
        auto temp2 = (uint32_t*)(tiling_.GetTiling());

        constexpr uint32_t tCubeTilingSize = ConstCeil(sizeof(TCubeTiling), CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
        GlobalTensor<int64_t> tilingGlobal;
        for (int i = 0; i < tCubeTilingSize; i += CACHE_LINE_SIZE) {
            Barrier();
            tilingGlobal.SetGlobalBuffer((__gm__ int64_t *)(msg->tilingInfo.tilingAddr + i));
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(tilingGlobal);
        }

        for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, temp1++, temp2++) {
            *temp2 = *temp1;
        }
        mul.Init(this->tiling_.GetTiling(), nullptr);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetOrgShape(__gm__ KfcMsg* msg)
{
    if constexpr (!ToMatmulConfig(MM_CFG).enableInit) {
        if (mul.GetSubBlockIdx() == 0) {
            msgAux.msg0.orgM = msg->orgShape.orgM;
            msgAux.msg0.orgN = msg->orgShape.orgN;
            msgAux.msg0.orgKa = msg->orgShape.orgKa;
            msgAux.msg0.orgKb = msg->orgShape.orgKb;
            msgAux.msg0.orgKc = msg->orgShape.orgKc;
            msgAux.msg0.setOrgShape = true;
        } else {
            msgAux.msg1.orgM = msg->orgShape.orgM;
            msgAux.msg1.orgN = msg->orgShape.orgN;
            msgAux.msg1.orgKa = msg->orgShape.orgKa;
            msgAux.msg1.orgKb = msg->orgShape.orgKb;
            msgAux.msg1.orgKc = msg->orgShape.orgKc;
            msgAux.msg1.setOrgShape = true;
        }
    } else {
        mul.SetOrgShape(msg->orgShape.orgM, msg->orgShape.orgN, msg->orgShape.orgKa, msg->orgShape.orgKb,
            msg->orgShape.orgKc);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorA(__gm__ KfcMsg* msg)
{
    if (!msg->body.setTensorA)
        return;
    if constexpr (A_TYPE::format == CubeFormat::SCALAR) {
        SrcAT scalar;
        auto temp1 = reinterpret_cast<__gm__ uint8_t*>(&(msg->body.aAddr));
        auto temp2 = (uint8_t*)&scalar;

        for (int i = 0; i < sizeof(SrcAT); i++, temp1++, temp2++) {
            *temp2 = *temp1;
        }
        mul.SetTensorA(scalar);
        return;
    }
    const uint64_t size = (uint64_t)(msg->body.sizeAmatrix);
    if constexpr (PhyPosIsL1(A_TYPE::pos)) {
        const auto& scmLocal = GetTscmTensor<typename A_TYPE::T>(msg->body.aAddr, size);
        mul.SetTensorA(scmLocal, msg->body.isTransA);
    } else {
        GlobalTensor<SrcAT> aGlobal;
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcAT*>(msg->body.aAddr), size);
        mul.SetTensorA(aGlobal, msg->body.isTransA);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorA(
    __gm__ KfcMsg* msg, const uint64_t size, const uint64_t offset)
{
    if (!msg->body.setTensorA) {
        return;
    }
    if constexpr (A_TYPE::format == CubeFormat::SCALAR) {
        SrcAT scalar;
        auto temp1 = reinterpret_cast<__gm__ uint8_t*>(&(msg->body.aAddr) + offset);
        auto temp2 = (uint8_t*)&scalar;

        for (int i = 0; i < sizeof(SrcAT); i++, temp1++, temp2++) {
            *temp2 = *temp1;
        }
        mul.SetTensorA(scalar);
        return;
    }
    if constexpr (PhyPosIsL1(A_TYPE::pos)) {
        const auto& scmLocal = GetTscmTensor<typename A_TYPE::T>(msg->body.aAddr + offset, size);
        mul.SetTensorA(scmLocal, msg->body.isTransA);
    } else {
        GlobalTensor<SrcAT> aGlobal;
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcAT*>(msg->body.aAddr + offset), size);
        mul.SetTensorA(aGlobal, msg->body.isTransA);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorB(__gm__ KfcMsg* msg)
{
    if (!msg->body.setTensorB)
        return;
    if constexpr (B_TYPE::format == CubeFormat::SCALAR) {
        SrcBT scalar;
        auto temp1 = reinterpret_cast<__gm__ uint8_t*>(&(msg->body.bAddr));
        auto temp2 = (uint8_t*)&scalar;

        for (int i = 0; i < sizeof(SrcBT); i++, temp1++, temp2++) {
            *temp2 = *temp1;
        }
        mul.SetTensorB(scalar);
        return;
    }
    const uint64_t size = (uint64_t)(msg->body.sizeBmatrix);
    if constexpr (PhyPosIsL1(B_TYPE::pos)) {
        const auto& scmLocal = GetTscmTensor<typename B_TYPE::T>(msg->body.bAddr, size);
        mul.SetTensorB(scmLocal, msg->body.isTransB);
    } else {
        GlobalTensor<SrcBT> bGlobal;
        bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcBT*>(msg->body.bAddr), size);
        mul.SetTensorB(bGlobal, msg->body.isTransB);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorB(
    __gm__ KfcMsg* msg, const uint64_t size, const uint64_t offset)
{
    if (!msg->body.setTensorB) {
        return;
    }
    if constexpr (B_TYPE::format == CubeFormat::SCALAR) {
        SrcBT scalar;
        auto temp1 = reinterpret_cast<__gm__ uint8_t*>(&(msg->body.bAddr) + offset);
        auto temp2 = (uint8_t*)&scalar;

        for (int i = 0; i < sizeof(SrcBT); i++, temp1++, temp2++) {
            *temp2 = *temp1;
        }
        mul.SetTensorB(scalar);
        return;
    }
    if constexpr (PhyPosIsL1(B_TYPE::pos)) {
        const auto& scmLocal = GetTscmTensor<typename B_TYPE::T>(msg->body.bAddr + offset, size);
        mul.SetTensorB(scmLocal, msg->body.isTransB);
    } else {
        GlobalTensor<SrcBT> bGlobal;
        bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcBT*>(msg->body.bAddr + offset), size);
        mul.SetTensorB(bGlobal, msg->body.isTransB);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetBias(__gm__ KfcMsg* msg)
{
    if (msg->body.setTensorBias) {
        const uint64_t size = (uint64_t)tiling_.GetSingleCoreN();
        if constexpr (PhyPosIsL1(BIAS_TYPE::pos)) {
            const auto& scmLocal = GetTscmTensor<typename BIAS_TYPE::T>(msg->body.biasAddr, size);
            mul.SetBias(scmLocal);
        } else {
            GlobalTensor<typename BIAS_TYPE::T> biasGlobal;
            biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ typename BIAS_TYPE::T*>(msg->body.biasAddr), size);
            mul.SetBias(biasGlobal);
        }
    } else {
        mul.DisableBias();
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetBias(
    __gm__ KfcMsg* msg, const uint64_t offset)
{
    if (msg->body.setTensorBias) {
        const uint64_t size = (uint64_t)tiling_.GetSingleCoreN();
        if constexpr (PhyPosIsL1(BIAS_TYPE::pos)) {
            const auto& scmLocal = GetTscmTensor<typename BIAS_TYPE::T>(msg->body.biasAddr + offset, size);
            mul.SetBias(scmLocal);
        } else {
            GlobalTensor<typename BIAS_TYPE::T> biasGlobal;
            biasGlobal.SetGlobalBuffer(
                reinterpret_cast<__gm__ typename BIAS_TYPE::T*>(msg->body.biasAddr + offset), size);
            mul.SetBias(biasGlobal);
        }
    } else {
        mul.DisableBias();
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline bool MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::GetTensorC(__gm__ KfcMsg* msg)
{
    if constexpr (A_TYPE::layout != LayoutMode::NONE) {
        return true;
    }
    uint64_t size;
    if constexpr (ToMatmulConfig(MM_CFG).baseMN != 0) {
        size = ToMatmulConfig(MM_CFG).baseMN;
    } else {
        size = tiling_.GetBaseM() * tiling_.GetBaseN();
    }
    if constexpr (PhyPosIsL1(C_TYPE::pos)) {
        const auto& scmLocal = GetTscmTensor<typename C_TYPE::T>(msg->body.cAddr, size);
        mul.GetTensorC(scmLocal, (uint8_t)(msg->body.enAtomic), msg->body.enSequentialWrite);
    } else {
        GlobalTensor<DstT> cGlobal;

        cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(msg->body.cAddr), size);
        mul.GetTensorC(cGlobal, (uint8_t)(msg->body.enAtomic), msg->body.enSequentialWrite);
    }
    // Now release UB
    if constexpr (PhyPosIsUB(C_TYPE::pos)) {
        if (unlikely(msg->ubAddr >= 0)) {
            kfcCommSrv->FreeUB(msg->ubAddr);
        }
    }
    if (msg->body.sync == 1) { // Synchronize
        uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + mul.GetSubBlockIdx());
        NotifyEvent<PIPE_FIX>(eventID);
    }
    return false;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline bool MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::IterateBatch(__gm__ KfcMsg* msg)
{
    if constexpr (A_TYPE::layout == LayoutMode::NONE) {
        return true;
    }
    // In the batch scenario, messages occupy 128 bytes. After the update, messages occupy 64 bytes.
    GlobalTensor<int64_t> msgGlobalTensor;
    msgGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(msg) + sizeof(int64_t));
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(msgGlobalTensor);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    if (msg->body.setQuant == 1) {
        ASSERT(msg->body.quantMode != 1);  // scalar mode is not supported for quantization parameters in
        // Batch MM
    }
#endif
    IterateSetMessage(msg);
    uint64_t size = tiling_.GetSingleCoreM() * tiling_.GetSingleCoreN();

    GlobalTensor<DstT> cGlobal;
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(msg->body.cAddr), size);
    mul.IterateBatch(cGlobal, msg->body.enPartialSum, (uint8_t)(msg->body.enAtomic),
        msg->body.enSequentialWrite, msg->body.matrixStrideA,
        msg->body.matrixStrideB, msg->body.matrixStrideC);

    // Now release UB
    if constexpr (PhyPosIsUB(A_TYPE::pos) || PhyPosIsUB(B_TYPE::pos) ||
        PhyPosIsUB(BIAS_TYPE::pos) || PhyPosIsUB(C_TYPE::pos)) {
        if (unlikely(msg->ubAddr >= 0)) {
            kfcCommSrv->FreeUB(msg->ubAddr);
        }
    }
    if (msg->body.sync || msg->body.waitIterateBatch) {
        uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + mul.GetSubBlockIdx());
        NotifyEvent<PIPE_FIX>(eventID);
    }
    return true;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::StartIterateNBatch(
    __gm__ KfcMsg* msg, uint32_t &cntIterator)
{
    const uint64_t size = tiling_.GetSingleCoreM() * tiling_.GetSingleCoreN();
    const uint64_t singleBatchASize = (uint64_t)(msg->body.sizeAmatrix) / msg->body.batchLoop;
    uint64_t batchAOffset = tiling_.GetALayoutInfoD() * msg->body.batchA;
    if constexpr (A_TYPE::layout != LayoutMode::SBNGD) {
        batchAOffset = batchAOffset * tiling_.GetALayoutInfoS();
    }
    const uint64_t singleBatchBSize = (uint64_t)(msg->body.sizeBmatrix) / msg->body.batchLoop;
    uint64_t batchBOffset = tiling_.GetBLayoutInfoD() * msg->body.batchB;
    if constexpr (B_TYPE::layout != LayoutMode::SBNGD) {
        batchBOffset = batchBOffset * tiling_.GetBLayoutInfoS();
    }
    const uint64_t batchCOffset = tiling_.GetCLayoutInfoS2();
    const uint32_t batchC = msg->body.batchA > msg->body.batchB ? msg->body.batchA : msg->body.batchB;
    bool layoutGCondition = tiling_.GetCLayoutInfoG() == 1 &&
                            (tiling_.GetBLayoutInfoG() != 1 || tiling_.GetALayoutInfoG() != 1);
    int32_t layoutG = tiling_.GetBLayoutInfoG() > tiling_.GetALayoutInfoG() ? tiling_.GetBLayoutInfoG() : tiling_.GetALayoutInfoG();
    int32_t batchOffsetBias = tiling_.GetCLayoutInfoS2() * batchC;
    if (layoutGCondition) {
        batchOffsetBias = batchOffsetBias / layoutG;
    }
    int32_t batchOffsetC = batchOffsetBias * sizeof(typename C_TYPE::T);
    if constexpr (C_TYPE::layout != LayoutMode::SBNGD) {
        batchOffsetC = batchOffsetC * tiling_.GetCLayoutInfoS1();
    }
    uint64_t offset = 0;
    for (uint32_t loopIdx = 0U; loopIdx < msg->body.batchLoop; loopIdx++) {
        const uint64_t aOffset = batchAOffset * loopIdx * sizeof(typename A_TYPE::T);
        const uint64_t bOffset = batchBOffset * loopIdx * sizeof(typename B_TYPE::T);
        const uint64_t biasOffset = batchOffsetBias * loopIdx * sizeof(typename BIAS_TYPE::T);
        IterateSetMessage(msg, singleBatchASize, singleBatchBSize, aOffset, bOffset, biasOffset);
        GlobalTensor<DstT> cGlobal;
        cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(msg->body.cAddr + offset), size);
        mul.IterateBatch(cGlobal, msg->body.enPartialSum, (uint8_t)(msg->body.enAtomic),
            msg->body.enSequentialWrite, msg->body.matrixStrideA,
            msg->body.matrixStrideB, msg->body.matrixStrideC);
        cntIterator++;
        if (cntIterator < INC_PROCESS_CHECK && (!msg->body.sync && !msg->body.waitIterateBatch)) {
            uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + mul.GetSubBlockIdx());
            NotifyEvent<PIPE_FIX>(eventID);
        }
        offset += batchOffsetC;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline bool MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::IterateNBatch(__gm__ KfcMsg* msg)
{
    if constexpr (!ToMatmulConfig(MM_CFG).isNBatch) {
        return true;
    }
    GlobalTensor<int64_t> msgGlobal;
    msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(msg) + sizeof(int64_t));
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(msgGlobal);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    if (msg->body.setQuant == 1) {
        ASSERT(msg->body.quantMode != 1);  // scalar mode is not supported for quantization parameters in
        // Batch MM
    }
#endif
    uint32_t cntIterator = 0;
    StartIterateNBatch(msg, cntIterator);
    // Now release UB
    if constexpr (PhyPosIsUB(A_TYPE::pos) || PhyPosIsUB(B_TYPE::pos) ||
        PhyPosIsUB(BIAS_TYPE::pos) || PhyPosIsUB(C_TYPE::pos)) {
        if (unlikely(msg->ubAddr >= 0)) {
            kfcCommSrv->FreeUB(msg->ubAddr);
        }
    }
    uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + mul.GetSubBlockIdx());
    if (msg->body.sync || msg->body.waitIterateBatch) {
        NotifyEvent<PIPE_FIX>(eventID);
    } else if (cntIterator >= INC_PROCESS_CHECK) {
        NotifyEvent<PIPE_FIX>(eventID);
    }
    return true;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::GetOffsetSize(
    __gm__ KfcMsg* msg, KFC_Enum funID, uint32_t sync, uint64_t &offsetSize, uint32_t &enSequentialWrite)
{
    if constexpr ((ToMatmulConfig(MM_CFG).iterateMode & IterateMode::ITERATE_MODE_NORMAL) == 0) {
        ASSERT(msg->body.cAddr != 0); // The output address must be configured.
        if constexpr (ToMatmulConfig(MM_CFG).baseMN != 0) {
            offsetSize = enSequentialWrite ? ToMatmulConfig(MM_CFG).baseMN : 0;
        } else {
            offsetSize = enSequentialWrite ? (tiling_.GetBaseM() * tiling_.GetBaseN()) : 0;
        }
    } else {
        if (funID == KFC_Enum::MMFUN_ITERATE_ALL) {
            ASSERT(msg->body.cAddr != 0); // The output address must be configured.
            if constexpr (ToMatmulConfig(MM_CFG).baseMN != 0) {
                offsetSize = enSequentialWrite ? ToMatmulConfig(MM_CFG).baseMN : 0;
            } else {
                offsetSize = enSequentialWrite ? (tiling_.GetBaseM() * tiling_.GetBaseN()) : 0;
            }
        } else if (sync == 0) {
            // For asynchronous Iterate, the offset must be used for address calculation and
            // the size is baseM x baseN.
            if constexpr (ToMatmulConfig(MM_CFG).baseMN != 0) {
                offsetSize = ToMatmulConfig(MM_CFG).baseMN;
            } else {
                offsetSize = tiling_.GetBaseM() * tiling_.GetBaseN();
            }
            enSequentialWrite = 1;
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline bool MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::StartIterate(
    __gm__ KfcMsg* msg, KFC_Enum funID, uint32_t sync, uint32_t &cntIterator)
{
    uint64_t size;
    if constexpr (ToMatmulConfig(MM_CFG).singleCoreMN != 0) {
        size = ToMatmulConfig(MM_CFG).singleCoreMN;
    } else {
        size = tiling_.GetSingleCoreM() * tiling_.GetSingleCoreN();
    }

    GlobalTensor<DstT> cGlobal;
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(msg->body.cAddr), size);
    const auto& scmLocal = GetTscmTensor<typename C_TYPE::T>(msg->body.cAddr, size);
    uint64_t offset = 0;
    uint64_t offsetSize = 0;
    auto enSequentialWrite = msg->body.enSequentialWrite;
    auto enAtomic = msg->body.enAtomic;
    auto enPartialSum = msg->body.enPartialSum;
    GetOffsetSize(msg, funID, sync, offsetSize, enSequentialWrite);
    TRACE_START(TraceId::MatMul_CALC);
    // Asynchronous and configure the workspace
    while (mul.Iterate(enPartialSum)) {
        if constexpr ((ToMatmulConfig(MM_CFG).iterateMode & IterateMode::ITERATE_MODE_NORMAL) != 0) {
            if (unlikely(cntIterator == 0)) {
                if (unlikely(funID == KFC_Enum::MMFUN_ITERATE && sync == 1)) {
                    TRACE_STOP(TraceId::MatMul_CALC);
                    return false; // The queue is not switched, and no message needs to be returned.
                }
            }
        }
        if constexpr (PhyPosIsL1(C_TYPE::pos)) {
            mul.GetTensorC(scmLocal[offset], (uint8_t)(enAtomic), enSequentialWrite);
        } else {
            mul.GetTensorC(cGlobal[offset], (uint8_t)(enAtomic), enSequentialWrite);
        }
        cntIterator++;
        if constexpr ((ToMatmulConfig(MM_CFG).iterateMode & IterateMode::ITERATE_MODE_NORMAL) != 0) {
            if (cntIterator < INC_PROCESS_CHECK) {
                if (funID == KFC_Enum::MMFUN_ITERATE) {
                    uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + mul.GetSubBlockIdx());
                    NotifyEvent<PIPE_FIX>(eventID);
                }
            }
        }
        offset += offsetSize;
    }
    return true;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline bool MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::Iterate(
    __gm__ KfcMsg* msg, KFC_Enum funID)
{
    if constexpr (A_TYPE::layout != LayoutMode::NONE) {
        return true;
    }
    if constexpr ((A_TYPE::ibShare == true) || (B_TYPE::ibShare == true)) {
        if (msg->body.iterateFakeMsg) {
            if (funID == KFC_Enum::MMFUN_ITERATE_ALL) { // fake msg
                uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + kfcCommSrv->subBlockID);
                NotifyEvent<PIPE_FIX>(eventID);
                return true;
            }
        }
    } else {
        ASSERT(!msg->body.iterateFakeMsg &&"Only Ib share mode support fake msg.");
    }
    SyncCubeWithVec<A_TYPE::ibShare, B_TYPE::ibShare>();
    QuantCacheRefresh(msg);
    IterateSetMessage(msg);
    uint32_t cntIterator = 0;
    auto sync = msg->body.sync;
    if(!StartIterate(msg, funID, sync, cntIterator)) {
        return false;
    }
    // Now release UB
    if constexpr (PhyPosIsUB(A_TYPE::pos) || PhyPosIsUB(B_TYPE::pos) ||
        PhyPosIsUB(BIAS_TYPE::pos) || PhyPosIsUB(C_TYPE::pos)) {
        if (unlikely(msg->ubAddr >= 0)) {
            kfcCommSrv->FreeUB(msg->ubAddr);
        }
    }

    uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + mul.GetSubBlockIdx());
    if (sync || msg->body.waitIterateAll) {
        ASSERT(funID == KFC_Enum::MMFUN_ITERATE_ALL);
        NotifyEvent<PIPE_FIX>(eventID);
    } else if (cntIterator >= INC_PROCESS_CHECK && funID == KFC_Enum::MMFUN_ITERATE) {
        NotifyEvent<PIPE_FIX>(eventID);
    }
    mul.End();
    TRACE_STOP(TraceId::MatMul_CALC);
    return true;
}
} // namespace AscendC
#endif // __MATMUL_SERVER_H__