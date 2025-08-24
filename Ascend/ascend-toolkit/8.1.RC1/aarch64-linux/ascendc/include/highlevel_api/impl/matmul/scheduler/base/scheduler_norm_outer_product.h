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
 * \file scheduler_norm_outer_product.h
 * \brief
 */
#ifndef IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_NORM_OUTER_PRODUCT_H
#define IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_NORM_OUTER_PRODUCT_H

#include "scheduler_intf.h"
#include "scheduler_norm_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    MatmulScheduler is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MatmulScheduler is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
    TriangularMode TR_MODE>
class MatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TR_MODE,
    enable_if_t<(isNormEnableScheduler<A_TYPE, MM_CFG> ||
    (A_TYPE::layout == LayoutMode::NORMAL && ToMatmulConfig(MM_CFG).batchMode == BatchMode::SINGLE_LARGE_THAN_L1))
    && MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()>> :
    public MatmulNormSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TR_MODE>
{
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(CopyCubeOut);
    MATMUL_USE_MODULE(MatmulUnitFlag);
    MATMUL_USE_MODULE(LoadToA2);
    MATMUL_USE_MODULE(LoadToB2);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(MmadCompute);

    using TransT = typename A_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using DstT = typename C_TYPE::T;
    using L0cT = typename GetDstType<typename A_TYPE::T>::Type;

public:
    using BASE_MODULE =
        AscendC::Impl::Detail::MatmulNormSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TR_MODE>;

    __aicore__ inline bool ScheduleOnce(bool enPartialSum)
    {
        if (!BASE_MODULE::MoveNext()) {
            return false;
        }
        if (!enPartialSum) {
            MATMUL_MODULE(CubeOutBuffer)->AllocTensor();
        }
        MATMUL_MODULE(BiasScheduler)->SetBias(!enPartialSum && MATMUL_MODULE(BiasScheduler)->IsBias());
        Compute(enPartialSum);
        return true;
    }

    __aicore__ inline void Compute(bool enPartialSum)
    {
        if constexpr(ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_N) {
            ComputeMultiIterOrderN(enPartialSum);
        } else {
            ComputeMultiIterOrderM(enPartialSum);
        }
    }

    __aicore__ inline void GetResult(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        GetResultImpl(gm, enAtomic, enSequentialWrite);
    }

    __aicore__ inline void GetResult(const LocalTensor<DstT>& co2Local, uint8_t enAtomic = 0,
        bool enSequentialWrite = false)
    {
        static_assert(ToMatmulConfig(MM_CFG).scheduleType != ScheduleType::OUTER_PRODUCT,
            "When scheduleType is OUTER_PRODUCT, GetResult function of LocalTensor output is not supported.");
    }

private:
    __aicore__ inline void ComputeMultiIterOrderN(bool enPartialSum)
    {
        // init split params for left and right matrix
        SplitParams aL0Params = BASE_MODULE::InitSplitAParams();
        SplitParams bL0Params = BASE_MODULE::InitSplitBParams();

        // start K outer loop
        MATMUL_MODULE(KLoop)->OuterStart();
        do {
            int32_t kOuterIdx = MATMUL_MODULE(KLoop)->GetOuterIdx();
            // CopyIn
            LocalTensor<TransT> b1 = MATMUL_MODULE(CopyCubeInB)->LoadData(
                kOuterIdx, MATMUL_MODULE(NLoop)->GetInnerIdx(),
                MATMUL_MODULE(KLoop)->GetTileShapeB(), MATMUL_MODULE(NLoop)->GetTileShape());
            // update some params in SplitParams which is related to k loop
            BASE_MODULE::UpdateSplitParams(aL0Params, bL0Params);
            SplitBias(bL0Params.axisL0Len);
            bool isATranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
            bool isBTranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();
            MATMUL_MODULE(LoadToB2)->Prepare(isBTranspose, bL0Params.kAxisL1Len);
            ComputeLoopM(b1, aL0Params, bL0Params, enPartialSum, kOuterIdx, isATranspose, isBTranspose);
            MATMUL_MODULE(BiasScheduler)->Free();

            MATMUL_MODULE(CopyCubeInB)->ClearLoadData(b1, kOuterIdx, MATMUL_MODULE(NLoop)->GetInnerIdx());
        } while (MATMUL_MODULE(KLoop)->OuterNext());
    }

    __aicore__ inline void ComputeLoopM(const LocalTensor<TransT>& b1, const SplitParams& aL0Params,
        const SplitParams& bL0Params, const bool enPartialSum, const int32_t kOuterIdx, const bool isATranspose, const bool isBTranspose)
    {
        bool sL0CInit = false;
        bool sL0CLast = false;
        BASE_MODULE::UpdateComputeParams(enPartialSum, sL0CInit, sL0CLast);
        uint32_t l0aDBLoop = MATMUL_MODULE(MLoop)->GetL0DBLoopNum();
        // load b2 from L1 to L0
        int32_t kL0Len = MATMUL_MODULE(KLoop)->GetTileShapeA();
        int dbUsedM = MATMUL_MODULE(MLoop)->GetTileShape();
        int mL1Len = aL0Params.axisL1Len;
        auto madMLen = aL0Params.axisL0Len;
        auto posB = MATMUL_MODULE(NLoop)->GetInnerIdx() * MATMUL_MODULE(KLoop)->GetTotalIter() + MATMUL_MODULE(KLoop)->GetInnerIdx();
        for (int dbInner = 0; dbInner < l0aDBLoop; dbInner++) {
            if (dbInner > 0) {
                if (MATMUL_MODULE(MLoop)->GetInnerIdx() + DB_NUM == MATMUL_MODULE(MLoop)->GetTotalIter()) {
                    dbUsedM =  MATMUL_MODULE(MLoop)->GetTailShape();
                    mL1Len = CeilAlign(dbUsedM, BLOCK_CUBE);
                    madMLen = dbUsedM;
                    madMLen = BASE_MODULE::GetFixedMadM(madMLen);
                }
            }
            LocalTensor<TransT> a1 = MATMUL_MODULE(CopyCubeInA)->LoadData(MATMUL_MODULE(MLoop)->GetInnerIdx() + dbInner,
                kOuterIdx, dbUsedM, MATMUL_MODULE(KLoop)->GetTileShapeA());
            // allocate L0 buffer
            auto& bufferPool = MATMUL_MODULE(TBufPoolL0)->Allocate();
            auto posA = (MATMUL_MODULE(MLoop)->GetInnerIdx() + dbInner) * MATMUL_MODULE(KLoop)->GetTotalIter() +
                MATMUL_MODULE(KLoop)->GetInnerIdx();
            LocalTensor<TransT> a2 = SplitA(a1, aL0Params, posA, mL1Len, madMLen, kL0Len, isATranspose);
            LocalTensor<TransT> b2 = SplitB(b1, bL0Params, posB, bL0Params.axisL1Len, bL0Params.axisL0Len, kL0Len, isBTranspose);
            bufferPool.EnQue();
            bufferPool.DeQue();
            // prepare params and compute
            CubeCompute(MATMUL_MODULE(CubeOutBuffer)->GetTensor()[
                dbInner * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() *
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN()], a2, b2, madMLen, bL0Params.axisL0Len,
                kL0Len, isATranspose, isBTranspose, enPartialSum, sL0CInit, sL0CLast);
            bufferPool.Free();
            MATMUL_MODULE(CopyCubeInA)->ClearLoadData(a1, MATMUL_MODULE(MLoop)->GetInnerIdx() + dbInner, kOuterIdx);
        }
    }

    __aicore__ inline void ComputeMultiIterOrderM(bool enPartialSum)
    {
        // init split params for left and right matrix
        SplitParams aL0Params = BASE_MODULE::InitSplitAParams();
        SplitParams bL0Params = BASE_MODULE::InitSplitBParams();

        // start K outer loop
        MATMUL_MODULE(KLoop)->OuterStart();
        do {
            int32_t kOuterIdx = MATMUL_MODULE(KLoop)->GetOuterIdx();
            // CopyIn
            LocalTensor<TransT> a1 = MATMUL_MODULE(CopyCubeInA)->LoadData(MATMUL_MODULE(MLoop)->GetInnerIdx(),
                kOuterIdx, MATMUL_MODULE(MLoop)->GetTileShape(), MATMUL_MODULE(KLoop)->GetTileShapeA());
            // update some params in SplitParams which is related to k loop
            auto bias = SplitBias();
            bool isATranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
            bool isBTranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();
            BASE_MODULE::SplitPrepare(isATranspose, isBTranspose, aL0Params, bL0Params);
            ComputeLoopN(a1, bias, aL0Params, bL0Params, enPartialSum, kOuterIdx, isATranspose, isBTranspose);
            MATMUL_MODULE(BiasScheduler)->Free(bias);
            MATMUL_MODULE(CopyCubeInA)->ClearLoadData(a1, MATMUL_MODULE(MLoop)->GetInnerIdx(), kOuterIdx);
        } while (MATMUL_MODULE(KLoop)->OuterNext());
    }

    __aicore__ inline void ComputeLoopN(const LocalTensor<TransT>& a1, LocalTensor<BiasT>& bias, const SplitParams& aL0Params,
        const SplitParams& bL0Params, const bool enPartialSum, const int32_t kOuterIdx, const bool isATranspose, const bool isBTranspose)
    {
        bool sL0CInit = false;
        bool sL0CLast = false;
        BASE_MODULE::UpdateComputeParams(enPartialSum, sL0CInit, sL0CLast);

        uint32_t l0bDBLoop = MATMUL_MODULE(NLoop)->GetL0DBLoopNum();
        // load b2 from L1 to L0
        int32_t kL0Len = MATMUL_MODULE(KLoop)->GetTileShapeA();
        // Split b2
        int dbUsedN = MATMUL_MODULE(NLoop)->GetTileShape();
        int nL1Len = bL0Params.axisL1Len;
        int madNLen = bL0Params.axisL0Len;
        auto posA = MATMUL_MODULE(MLoop)->GetInnerIdx() * MATMUL_MODULE(KLoop)->GetTotalIter() + MATMUL_MODULE(KLoop)->GetInnerIdx();
        for (int dbInner = 0; dbInner < l0bDBLoop; dbInner++) {
            if (dbInner > 0) {
                if (MATMUL_MODULE(NLoop)->GetInnerIdx() + DB_NUM == MATMUL_MODULE(NLoop)->GetTotalIter()) {
                    dbUsedN = MATMUL_MODULE(NLoop)->GetTailShape();
                    nL1Len = CeilAlign(dbUsedN, BLOCK_CUBE);
                    madNLen = dbUsedN;
                }
            }
            LocalTensor<TransT> b1 = MATMUL_MODULE(CopyCubeInB)->LoadData(kOuterIdx,
                MATMUL_MODULE(NLoop)->GetInnerIdx() + dbInner, MATMUL_MODULE(KLoop)->GetTileShapeB(), dbUsedN);

            // allocate L0 buffer
            auto& bufferPool = MATMUL_MODULE(TBufPoolL0)->Allocate();
            auto posB = (MATMUL_MODULE(NLoop)->GetInnerIdx() + dbInner) * MATMUL_MODULE(KLoop)->GetTotalIter() +
                MATMUL_MODULE(KLoop)->GetInnerIdx();
            LocalTensor<TransT> a2 = SplitA(a1, aL0Params, posA, aL0Params.axisL1Len, aL0Params.axisL0Len, kL0Len, isATranspose);
            LocalTensor<TransT> b2 = SplitB(b1, bL0Params, posB, nL1Len, madNLen, kL0Len, isBTranspose);
            bufferPool.EnQue();
            bufferPool.DeQue();
            // load bias
            MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, dbUsedN, dbInner * bL0Params.axisL0Len);
            // prepare params and compute
            CubeCompute(MATMUL_MODULE(CubeOutBuffer)->GetTensor()[
                dbInner * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() *
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN()],
                a2, b2, aL0Params.axisL0Len, madNLen, kL0Len, isATranspose, isBTranspose, enPartialSum, sL0CInit, sL0CLast);
            bufferPool.Free();
            MATMUL_MODULE(BiasScheduler)->Free();
            MATMUL_MODULE(CopyCubeInB)->ClearLoadData(b1, kOuterIdx, MATMUL_MODULE(NLoop)->GetInnerIdx() + dbInner);
        }
    }

    __aicore__ inline LocalTensor<BiasT> SplitBias(const int32_t dataLen = 0)
    {
        if constexpr (ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_N) {
            LocalTensor<BiasT> bias = MATMUL_MODULE(BiasScheduler)->CopyIn(MATMUL_MODULE(NLoop)->GetBaseShape(), 1,
                MATMUL_MODULE(NLoop)->GetInnerIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
            MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, dataLen);
            MATMUL_MODULE(BiasScheduler)->Free(bias);
            return {};
        } else {
             return MATMUL_MODULE(BiasScheduler)->CopyIn(MATMUL_MODULE(NLoop)->GetBaseShape(), 1,
                MATMUL_MODULE(NLoop)->GetInnerIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
        }
    }

    __aicore__ inline void CubeCompute(const LocalTensor<L0cT>& cMatrix, const LocalTensor<TransT>& a2,
        const LocalTensor<TransT>& b2, const uint16_t madM, const uint16_t madN, const uint16_t madK,
        const bool isATranspose, const bool isBTranspose, const bool enPartialSum, const bool sL0CInit, const bool sL0CLast)
    {
        uint8_t unitFlag = MATMUL_MODULE(MatmulUnitFlag)->GetUnitFlag(sL0CLast);
        bool isBias;
        bool cmatrixInitVal;
        bool cmatrixSource;
        BASE_MODULE::UpdateBiasParams(enPartialSum, sL0CInit, cmatrixSource, cmatrixInitVal, isBias);
        MATMUL_MODULE(MmadCompute)->Compute(cMatrix, a2, b2, madM, madK, madN,
            isATranspose, isBTranspose, unitFlag, cmatrixSource, cmatrixInitVal, isBias);
    }

    __aicore__ inline LocalTensor<TransT> SplitA(const LocalTensor<TransT>& a1, const SplitParams& aL0Params,
    const int32_t posA, const uint16_t mL1Len, const uint16_t madMLen, const uint16_t kL0Len, const bool isATranspose)
    {
        // Split
        LocalTensor<TransT> a2;
        if (!(MATMUL_MODULE(TBufPoolL0)->template Hit<TPosition::A2>(posA))) {
            a2 = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::A2, TransT>();
            if constexpr (ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_N) {
                MATMUL_MODULE(LoadToA2)->Prepare(isATranspose, aL0Params.kAxisL1Len, mL1Len);
            }
            MATMUL_MODULE(LoadToA2)->Load(a2, a1, mL1Len, aL0Params.kAxisL1Len,
                madMLen, kL0Len, aL0Params.axisL1Offset, aL0Params.kAxisL1Offset, isATranspose);
        } else {
            a2 = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::A2, TransT>();
        }
        return a2;
    }

    __aicore__ inline LocalTensor<TransT> SplitB(const LocalTensor<TransT>& b1, const SplitParams& bL0Params,
    const int32_t posB, const uint16_t nL1Len, const uint16_t madNLen, const uint16_t kL0Len, const bool isBTranspose)
    {
        LocalTensor<TransT> b2;
        if (!(MATMUL_MODULE(TBufPoolL0)->template Hit<TPosition::B2>(posB))) {
            b2 = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::B2, TransT>();
            MATMUL_MODULE(LoadToB2)->Load(b2, b1, nL1Len, bL0Params.kAxisL1Len,
                madNLen, kL0Len, bL0Params.axisL1Offset, bL0Params.kAxisL1Offset, isBTranspose);
        } else {
            b2 = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::B2, TransT>();
        }
        return b2;
    }

    __aicore__ inline void GetResultImpl(const GlobalTensor<DstT>& gm, uint8_t enAtomic, bool enSequentialWrite)
    {
        auto co1Local = MATMUL_MODULE(CubeOutBuffer)->GetTensor();
        MATMUL_MODULE(CubeOutBuffer)->EnQue(co1Local);
        MATMUL_MODULE(CubeOutBuffer)->DeQue();
        int32_t co1Offset = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() *
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        int32_t baseHeight = MATMUL_MODULE(MLoop)->GetBaseShape();
        int32_t baseWidth = MATMUL_MODULE(NLoop)->GetBaseShape();
        int32_t baseBlockHeight = MATMUL_MODULE(MLoop)->GetBaseBlockShape();
        int32_t baseBlockWidth = MATMUL_MODULE(NLoop)->GetBaseBlockShape();
        if (MATMUL_MODULE(NLoop)->GetL0DBLoopNum() > 1) { // Means L0 N db, need to excute twice FixpipeL0CToGm
            FixpipeL0CToGm(gm, co1Local, MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                enAtomic, enSequentialWrite, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
            if (MATMUL_MODULE(NLoop)->GetInnerIdx() + DB_NUM == MATMUL_MODULE(NLoop)->GetTotalIter()) {
                baseWidth = MATMUL_MODULE(NLoop)->GetTailShape();
                baseBlockWidth = Ceil(baseWidth, BLOCK_CUBE);
            }
            FixpipeL0CToGm(gm, co1Local[co1Offset],
                MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx() + 1,
                enAtomic, enSequentialWrite, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
        } else if (MATMUL_MODULE(MLoop)->GetL0DBLoopNum() > 1) { // Means L0 M db, need to excute twice FixpipeL0CToGm
            FixpipeL0CToGm(gm, co1Local,
                MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                enAtomic, enSequentialWrite, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
            if (MATMUL_MODULE(MLoop)->GetInnerIdx() + DB_NUM == MATMUL_MODULE(MLoop)->GetTotalIter()) {
                baseHeight = MATMUL_MODULE(MLoop)->GetTailShape();
                baseBlockHeight = Ceil(baseHeight, BLOCK_CUBE);
            }
            FixpipeL0CToGm(gm, co1Local[co1Offset],
                MATMUL_MODULE(MLoop)->GetInnerIdx() + 1, MATMUL_MODULE(NLoop)->GetInnerIdx(),
                enAtomic, enSequentialWrite, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
        } else {
            FixpipeL0CToGm(gm, co1Local, MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                enAtomic, enSequentialWrite, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
        }
        MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
    }

    __aicore__ inline void FixpipeL0CToGm(const GlobalTensor<DstT>& gm, const LocalTensor<L0cT>& co1Local,
        int32_t curM, int32_t curN, uint8_t enAtomic, bool enSequentialWrite,
        int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        if constexpr (C_TYPE::format != CubeFormat::ND && C_TYPE::format != CubeFormat::ND_ALIGN &&
            C_TYPE::format != CubeFormat::NZ) {
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR, "Data format of C matrix should be ND, ND_ALIGN or NZ."); });
        }
        BASE_MODULE::SetAtomic(enAtomic);
        if (enSequentialWrite) {
            MATMUL_MODULE(CopyCubeOut)->template Copy<true>(gm, co1Local, curM, curN,
                baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
        } else {
            MATMUL_MODULE(CopyCubeOut)->template Copy<false>(gm, co1Local, curM, curN,
                baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
        }

        BASE_MODULE::ClearAtomic(enAtomic);
    }

private:
    constexpr static int32_t DB_NUM = 2;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC

#endif // IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_NORM_OUTER_PRODUCT_H
