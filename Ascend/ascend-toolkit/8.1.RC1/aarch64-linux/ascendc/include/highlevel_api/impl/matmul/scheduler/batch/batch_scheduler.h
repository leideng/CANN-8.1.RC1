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
 * \file batch_scheduler.h
 * \brief
 */
#ifndef IMPL_MATMUL_SCHEDULER_BATCH_BATCH_SCHEDULER_H
#define IMPL_MATMUL_SCHEDULER_BATCH_BATCH_SCHEDULER_H

#include "batch_scheduler_intf.h"
#include "batch_scheduler_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    BatchScheduler is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    BatchScheduler is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG>
class BatchScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG,
    enable_if_t<!MatmulFeatureTrait<MM_CFG>::IsNeedUB() &&  DoMatmulNorm(MM_CFG) &&
    ((A_TYPE::layout != LayoutMode::NONE && ToMatmulConfig(MM_CFG).batchMode == BatchMode::BATCH_LESS_THAN_L1) ||
    (A_TYPE::layout == LayoutMode::NORMAL && ToMatmulConfig(MM_CFG).batchMode == BatchMode::BATCH_LARGE_THAN_L1))>>
    : public BatchSchedulerBase<IMPL,  A_TYPE,  B_TYPE,  C_TYPE,  BIAS_TYPE, MM_CFG>
{
    MATMUL_USE_MODULE(BatchLoop);
    MATMUL_USE_MODULE(BatchCopyCubeInA);
    MATMUL_USE_MODULE(BatchCopyCubeInB);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(MmadCompute);
    MATMUL_USE_MODULE(LoadToA2);
    MATMUL_USE_MODULE(LoadToB2);
    MATMUL_USE_MODULE(MatmulUnitFlag);

    using SrcT = typename A_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using DstT = typename C_TYPE::T;

public:
    // fix framework module name
    using BASE_MODULE = AscendC::Impl::Detail::BatchSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;

    __aicore__ inline BatchScheduler() = default;
    __aicore__ inline ~BatchScheduler() = default;

    __aicore__ inline void Schedule(const LocalTensor<DstT>& dst, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite,
        const uint32_t matrixStrideA, const uint32_t matrixStrideB, const uint32_t matrixStrideC)
    {}

    __aicore__ inline void Schedule(const GlobalTensor<DstT>& dst, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite,
        const uint32_t matrixStrideA, const uint32_t matrixStrideB, const uint32_t matrixStrideC)
    {
        // loop unrelated calculation
        MATMUL_MODULE(BiasScheduler)->SetBias(MATMUL_MODULE(BiasScheduler)->IsBias() && !enPartialSum);
        auto batchOffsetInfo = PrepareOffset();
        auto ctx = BASE_MODULE::PrepareContext();

        LocalTensor<BiasT> bias; // load bias to l1
        if constexpr (!ToMatmulConfig(MM_CFG).isBiasBatch) {
            bias = MATMUL_MODULE(BiasScheduler)->CopyIn(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreN());
        }

        const auto batchLoop = MATMUL_MODULE(BatchLoop);
        for (batchLoop->OuterStart(); !batchLoop->OuterEnd(); batchLoop->OuterNext()) {
            if constexpr (ToMatmulConfig(MM_CFG).isBiasBatch) {
                bias = MATMUL_MODULE(BiasScheduler)->CopyIn(
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreN(),
                    batchLoop->GetBatchNum(), batchLoop->GetBiasBatchSrcOffset());
            }

            auto a1 = MATMUL_MODULE(BatchCopyCubeInA)->AllocTensor();
            auto b1 = MATMUL_MODULE(BatchCopyCubeInB)->AllocTensor();
            event_t eventIDMte2ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
            event_t eventIDMToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
            auto batchLoop = MATMUL_MODULE(BatchLoop);
            for (batchLoop->SplitStart(); !batchLoop->SplitEnd(); batchLoop->SplitNext()) {
                MATMUL_MODULE(BatchCopyCubeInA)->BatchLoad(a1, matrixStrideA, batchLoop->GetOuterIndex(),
                    batchLoop->GetSplitIndex(), batchLoop->GetSplitSize());
                MATMUL_MODULE(BatchCopyCubeInB)->BatchLoad(b1, matrixStrideB, batchLoop->GetOuterIndex(),
                    batchLoop->GetSplitIndex(), batchLoop->GetSplitSize());
                SetFlag<HardEvent::MTE2_MTE1>(eventIDMte2ToMte1);
                WaitFlag<HardEvent::MTE2_MTE1>(eventIDMte2ToMte1);
                for (batchLoop->InnerStart(); !batchLoop->InnerEnd(); batchLoop->InnerNext()) {
                    BASE_MODULE::isFirstIter_ = true;
                    if (batchOffsetInfo.setBiasFlag && (batchLoop->GetBatchIndex() % batchOffsetInfo.divisorBias == 1)) {
                        MATMUL_MODULE(BiasScheduler)->StopBias(bias);
                    }
                    UpdateOffset(batchOffsetInfo, ctx); 
                    while (BASE_MODULE::MoveNext()) { // iterate
                        MATMUL_MODULE(CubeOutBuffer)->AllocTensor();
                        ComputeBatch(a1, b1, bias, enPartialSum, ctx);
                        BASE_MODULE::GetBatchResult(dst[batchLoop->GetDstOffset()], ctx, enAtomic,
                            enSequentialWrite);
                        SetFlag<HardEvent::M_MTE1>(eventIDMToMte1);
                        WaitFlag<HardEvent::M_MTE1>(eventIDMToMte1);
                    }
                }
                BASE_MODULE::End();
            }
            MATMUL_MODULE(BatchCopyCubeInA)->BatchDestroy();
            MATMUL_MODULE(BatchCopyCubeInB)->BatchDestroy();

            if constexpr (ToMatmulConfig(MM_CFG).isBiasBatch) {
                MATMUL_MODULE(BiasScheduler)->Destroy(bias);
            }
        }

        if constexpr (!ToMatmulConfig(MM_CFG).isBiasBatch) {
            MATMUL_MODULE(BiasScheduler)->Destroy(bias);
        }           
    }

private:
    __aicore__ inline BatchOffsetInfo PrepareOffset()
    {   
        // calculate corresponding mod, divisor, alignSize for A/B/Bias offset
        BatchOffsetInfo batchOffsetInfo;
        BASE_MODULE::CalcBatchIterateAOffsetInfo(batchOffsetInfo);
        BASE_MODULE::CalcBatchIterateBOffsetInfo(batchOffsetInfo);
        BASE_MODULE::CalcBatchIterateBiasOffsetInfo(batchOffsetInfo);
        return batchOffsetInfo;
    }

    __aicore__ inline void UpdateOffset(BatchOffsetInfo& batchOffsetInfo, BatchSchedulerContext& ctx)
    {
        auto batchIndex = MATMUL_MODULE(BatchLoop)->GetBatchIndex();
        ctx.offsetA = batchOffsetInfo.alignA *
            (batchIndex % batchOffsetInfo.modA + batchIndex / batchOffsetInfo.divisorA);
        ctx.offsetB = batchOffsetInfo.alignB *
            (batchIndex % batchOffsetInfo.modB + batchIndex / batchOffsetInfo.divisorB);
        ctx.offsetBias = batchOffsetInfo.alignBias *
            (batchIndex % batchOffsetInfo.modBias + batchIndex / batchOffsetInfo.divisorBias);
    }

    __aicore__ inline void ComputeBatch(LocalTensor<SrcT>& a1, LocalTensor<SrcT>& b1, LocalTensor<BiasT>& bias,
        bool enPartialSum, BatchSchedulerContext& ctx)
    {
        if constexpr (ToMatmulConfig(MM_CFG).scheduleType == ScheduleType::OUTER_PRODUCT){
            ComputeL0DB(a1, b1, bias, enPartialSum, ctx);
        } else if constexpr (IsBasic(MM_CFG)) {
            ComputeOneIter(a1, b1, bias, enPartialSum, ctx);
        } else {
            ComputeMultiIter(a1, b1, bias, enPartialSum, ctx);
        }
    }

    __aicore__ inline void ComputeMultiIter(LocalTensor<SrcT>& a1, LocalTensor<SrcT>& b1, LocalTensor<BiasT>& bias,
        bool enPartialSum, BatchSchedulerContext& ctx)
    {
        // init split params for left and right matrix (k loop unrelated)
        BASE_MODULE::InitSplitAParams(ctx.aL0Params);
        BASE_MODULE::InitSplitBParams(ctx.bL0Params);
        // start K outer loop
        MATMUL_MODULE(KLoop)->OuterStart();
        do {
            // update k outer loop related params
            int32_t sL0CInit;
            int32_t sL0CLast;
            BASE_MODULE::UpdateSplitParams(enPartialSum, ctx.aL0Params, ctx.bL0Params, sL0CInit, sL0CLast);
            // load bias to c2
            MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, ctx.bL0Params.axisL0Len,
                ctx.offsetBias + MATMUL_MODULE(NLoop)->GetOuterIdx() *
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
            BASE_MODULE::MacroCompute(a1, b1, ctx, sL0CInit, sL0CLast);
        } while (MATMUL_MODULE(KLoop)->OuterNext());
    }

    __aicore__ inline void ComputeOneIter(LocalTensor<SrcT>& a1, LocalTensor<SrcT>& b1, LocalTensor<BiasT>& bias,
        bool enPartialSum, BatchSchedulerContext& ctx)
    {
        // init split params for left and right matrix (k loop unrelated)
        BASE_MODULE::InitSplitAParams(ctx.aL0Params);
        BASE_MODULE::InitSplitBParams(ctx.bL0Params);
        // start k outer loop
        MATMUL_MODULE(KLoop)->OuterStart();
        // update k outer loop related params
        BASE_MODULE::UpdateSplitParams(ctx.aL0Params, ctx.bL0Params);
        int32_t sL0CInit = enPartialSum ? 0 : 1;
        // load bias to c2
        MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, ctx.bL0Params.axisL0Len, 
            ctx.offsetBias + MATMUL_MODULE(NLoop)->GetOuterIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
        BASE_MODULE::MacroCompute(a1, b1, ctx, sL0CInit, 1);
    }

     __aicore__ inline void ComputeL0DB(LocalTensor<SrcT>& a1, LocalTensor<SrcT>& b1, LocalTensor<BiasT>& bias,
        bool enPartialSum, BatchSchedulerContext& ctx)
    {
        BASE_MODULE::InitSplitAParams(ctx.aL0Params); // init Kloop unrelated params
        BASE_MODULE::InitSplitBParams(ctx.bL0Params);

        MATMUL_MODULE(KLoop)->OuterStart();
        MATMUL_MODULE(KLoop)->InnerStart();
        do {
            int32_t sL0CInit;
            int32_t sL0CLast;
            BASE_MODULE::UpdateSplitParams(enPartialSum, ctx.aL0Params, ctx.bL0Params, sL0CInit, sL0CLast);
            if constexpr (ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_N) {
                MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, ctx.bL0Params.axisL0Len,
                    ctx.offsetBias + MATMUL_MODULE(NLoop)->GetOuterIdx() *
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
            }
            MATMUL_MODULE(LoadToA2)->Prepare(MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA(),
                ctx.aL0Params.kAxisL1Len, ctx.aL0Params.axisL1Len);  // SingleCoreM
            MATMUL_MODULE(LoadToB2)->Prepare(MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB(),
                ctx.bL0Params.kAxisL1Len);
            if constexpr (ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_M) {
                ComputeMDb(a1, b1, bias, ctx, sL0CInit, sL0CLast, enPartialSum);
            } else {
                ComputeNDb(a1, b1, bias, ctx, sL0CInit, sL0CLast, enPartialSum);
            }
        } while(MATMUL_MODULE(KLoop)->OuterNext());
    }

    __aicore__ inline void ComputeNDb(LocalTensor<SrcT>& a1, LocalTensor<SrcT>& b1, LocalTensor<BiasT>& bias,
        BatchSchedulerContext& ctx, int32_t sL0CInit, int32_t sL0CLast, bool enPartialSum)
    {
        int32_t aKL1Offset = ctx.aL0Params.kAxisL1Offset; // k x baseK
        int32_t bKL1Offset = ctx.bL0Params.kAxisL1Offset;
        int32_t computeK = MATMUL_MODULE(KLoop)->GetBaseShape();
        int32_t nL0DBLoop = MATMUL_MODULE(MLoop)->GetL0DBLoopNum();

        LocalTensor<SrcT> b2 = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::B2, SrcT>();
        MATMUL_MODULE(LoadToB2)->Load(b2, b1[ctx.offsetB], ctx.bL0Params.axisL1Len,
            ctx.bL0Params.kAxisL1Len, ctx.bL0Params.axisL0Len,
            computeK,  ctx.bL0Params.axisL1Offset,
            bKL1Offset, MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB());
        int32_t axisL1DbOffset = ctx.aL0Params.axisL1Offset;
        for (int32_t idx = 0; idx < nL0DBLoop; ++idx) {
            auto& bufferPool = MATMUL_MODULE(TBufPoolL0)->Allocate();
            LocalTensor<SrcT> a2 = bufferPool.template GetBuffer<TPosition::A2, SrcT>();
            MATMUL_MODULE(LoadToA2)->Load(a2, a1[ctx.offsetA],
                ctx.aL0Params.axisL1Len, ctx.aL0Params.kAxisL1Len,
                ctx.aL0Params.axisL0Len, computeK, axisL1DbOffset, aKL1Offset,
                MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA());

            bufferPool.EnQue();
            bufferPool.DeQue();

            bool cmatrixInitVal;
            bool cmatrixSource;
            BASE_MODULE::UpdateMmadComputeParams(sL0CInit, cmatrixSource, cmatrixInitVal);
            MATMUL_MODULE(MmadCompute)->Compute(MATMUL_MODULE(CubeOutBuffer)->GetTensor()[idx *
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() *
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN()],
                    a2, b2, ctx.aL0Params.axisL0Len, computeK, ctx.bL0Params.axisL0Len,
                    MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA(),
                    MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB(),
                    MATMUL_MODULE(MatmulUnitFlag)->GetUnitFlag(sL0CLast),
                    cmatrixSource,
                    cmatrixInitVal,
                    false);
            bufferPool.Free();
            MATMUL_MODULE(BiasScheduler)->Free();
            axisL1DbOffset += MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
        }
    }

    __aicore__ inline void ComputeMDb(LocalTensor<SrcT>& a1, LocalTensor<SrcT>& b1, LocalTensor<BiasT>& bias,
        BatchSchedulerContext& ctx, int32_t sL0CInit, int32_t sL0CLast, bool enPartialSum)
    {
        int32_t aKL1Offset = ctx.aL0Params.kAxisL1Offset; // k x baseK
        int32_t bKL1Offset = ctx.bL0Params.kAxisL1Offset;
        int32_t computeK = MATMUL_MODULE(KLoop)->GetBaseShape();
        int32_t mL0DbLoop = MATMUL_MODULE(NLoop)->GetL0DBLoopNum();

        LocalTensor<SrcT> a2 = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::A2, SrcT>();
        MATMUL_MODULE(LoadToA2)->Load(a2, a1[ctx.offsetA],
            ctx.aL0Params.axisL1Len, ctx.aL0Params.kAxisL1Len, ctx.aL0Params.axisL0Len,
            computeK, ctx.aL0Params.axisL1Offset, aKL1Offset,
            MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA());

        int32_t axisL1DbOffset = ctx.bL0Params.axisL1Offset;
        for (int32_t idx = 0; idx < mL0DbLoop; ++idx) {
            auto& bufferPool = MATMUL_MODULE(TBufPoolL0)->Allocate();
            LocalTensor<SrcT> b2 = bufferPool.template GetBuffer<TPosition::B2, SrcT>();
            MATMUL_MODULE(LoadToB2)->Load(b2, b1[ctx.offsetB],
                ctx.bL0Params.axisL1Len, ctx.bL0Params.kAxisL1Len, ctx.bL0Params.axisL0Len,
                computeK, axisL1DbOffset, bKL1Offset,
                MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB());

            bufferPool.EnQue();
            bufferPool.DeQue();

            MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, ctx.bL0Params.axisL0Len,
                ctx.offsetBias + (MATMUL_MODULE(NLoop)->GetOuterIdx() + idx) *
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());

            bool cmatrixSource;
            bool cmatrixInitVal;
            BASE_MODULE::UpdateMmadComputeParams(sL0CInit, cmatrixSource, cmatrixInitVal);
            MATMUL_MODULE(MmadCompute)->Compute(MATMUL_MODULE(CubeOutBuffer)->GetTensor()[idx *
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() *
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN()],
                    a2, b2,
                    ctx.aL0Params.axisL0Len, computeK, ctx.bL0Params.axisL0Len,
                    MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA(),
                    MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB(),
                    MATMUL_MODULE(MatmulUnitFlag)->GetUnitFlag(sL0CLast),
                    cmatrixSource,
                    cmatrixInitVal,
                    false);
            bufferPool.Free();
            axisL1DbOffset += MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
            MATMUL_MODULE(BiasScheduler)->Free();
        }
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_SCHEDULER_BATCH_BATCH_SCHEDULER_H
