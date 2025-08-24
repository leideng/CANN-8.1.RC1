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
 * \file batch_scheduler_v200.h
 * \brief
 */
#ifndef IMPL_MATMUL_SCHEDULER_BATCH_BATCH_SCHEDULER_V200_H
#define IMPL_MATMUL_SCHEDULER_BATCH_BATCH_SCHEDULER_V200_H

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
class BatchScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, enable_if_t<
    MatmulFeatureTrait<MM_CFG>::IsNeedUB() && !MatmulFeatureTrait<MM_CFG>::IsSupportCmatrixInitVal() &&
    DoMatmulNorm(MM_CFG) &&
    ((A_TYPE::layout != LayoutMode::NONE && ToMatmulConfig(MM_CFG).batchMode == BatchMode::BATCH_LESS_THAN_L1) ||
    (A_TYPE::layout == LayoutMode::NORMAL && ToMatmulConfig(MM_CFG).batchMode == BatchMode::BATCH_LARGE_THAN_L1)) &&
    (ToMatmulConfig(MM_CFG).scheduleType != ScheduleType::OUTER_PRODUCT)>>
    : public BatchSchedulerBase<IMPL,  A_TYPE,  B_TYPE,  C_TYPE,  BIAS_TYPE, MM_CFG>
{
    MATMUL_USE_MODULE(BatchLoop);
    MATMUL_USE_MODULE(BatchCopyCubeInA);
    MATMUL_USE_MODULE(BatchCopyCubeInB);
    MATMUL_USE_MODULE(MatmulQuantProcessor);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(BiasScheduler);

    using SrcT = typename A_TYPE::T;
    using DstT = typename C_TYPE::T;

public:
    // fix framework module name
    using BASE_MODULE = AscendC::Impl::Detail::BatchSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;

    __aicore__ inline BatchScheduler() = default;
    __aicore__ inline ~BatchScheduler() = default;

    template <class T>
    __aicore__ inline void Schedule(const T& dst, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite,
        const uint32_t matrixStrideA, const uint32_t matrixStrideB, const uint32_t matrixStrideC)
    {
        if (A_TYPE::layout != LayoutMode::NORMAL) {
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BMM only support LayoutMode::NORMAL on 310P");});
        }

        // loop unrelated calculation
        MATMUL_MODULE(BiasScheduler)->SetBias(MATMUL_MODULE(BiasScheduler)->IsBias() && !enPartialSum);
        auto batchOffsetInfo = PrepareOffset();
        auto ctx = BASE_MODULE::PrepareContext();

        const auto batchLoop = MATMUL_MODULE(BatchLoop);
        for (batchLoop->OuterStart(); !batchLoop->OuterEnd(); batchLoop->OuterNext()) {
            GlobalTensor<uint64_t> global;
            global.SetGlobalBuffer((__gm__ uint64_t*)0);
            DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE>(global);

            auto a1 = MATMUL_MODULE(BatchCopyCubeInA)->AllocTensor();
            auto b1 = MATMUL_MODULE(BatchCopyCubeInB)->AllocTensor();
            event_t eventIDMte2ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
            event_t eventIDMToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
            for (batchLoop->SplitStart(); !batchLoop->SplitEnd(); batchLoop->SplitNext()) {
                MATMUL_MODULE(BatchCopyCubeInA)->BatchLoad(a1, matrixStrideA, batchLoop->GetOuterIndex(),
                    batchLoop->GetSplitIndex(), batchLoop->GetSplitSize());
                MATMUL_MODULE(BatchCopyCubeInB)->BatchLoad(b1, matrixStrideB, batchLoop->GetOuterIndex(),
                    batchLoop->GetSplitIndex(), batchLoop->GetSplitSize());
                SetFlag<HardEvent::MTE2_MTE1>(eventIDMte2ToMte1);
                WaitFlag<HardEvent::MTE2_MTE1>(eventIDMte2ToMte1);
                for (batchLoop->InnerStart(); !batchLoop->InnerEnd(); batchLoop->InnerNext()) {
                    if constexpr (IsSameTypeV<SrcT, int8_t> && IsSameTypeV<DstT, half>) {
                        if (batchLoop->GetInnerIndex() != 0 || batchLoop->GetSplitIndex() != 0) {
                            MATMUL_MODULE(MatmulQuantProcessor)->UpdateQuantTensor(
                                MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN());
                        }
                    }
                    BASE_MODULE::isFirstIter_ = true;
                    UpdateOffset(batchOffsetInfo, ctx);
                    while (BASE_MODULE::MoveNext()) {
                        MATMUL_MODULE(CubeOutBuffer)->AllocTensor();
                        if constexpr (IsBasic(MM_CFG)) {
                            ComputeOneIter(a1, b1, enPartialSum, ctx);
                        } else {
                            ComputeMultiIter(a1, b1, enPartialSum, ctx);
                        }
                        CopyOut(dst[batchLoop->GetDstOffset()], ctx, enAtomic, enSequentialWrite);
                    }
                }
                BASE_MODULE::End();
            }
            MATMUL_MODULE(BatchCopyCubeInA)->BatchDestroy();
            MATMUL_MODULE(BatchCopyCubeInB)->BatchDestroy();
        }
    }

private:
    __aicore__ inline BatchOffsetInfo PrepareOffset()
    {   
        // calculate corresponding mod, divisor, alignSize for A/B/Bias offset
        BatchOffsetInfo batchOffsetInfo;
        BASE_MODULE::CalcBatchIterateAOffsetInfo(batchOffsetInfo);
        BASE_MODULE::CalcBatchIterateBOffsetInfo(batchOffsetInfo);
        return batchOffsetInfo;
    }

    __aicore__ inline void UpdateOffset(BatchOffsetInfo& batchOffsetInfo, BatchSchedulerContext& ctx)
    {
        auto batchIndex = MATMUL_MODULE(BatchLoop)->GetBatchIndex();
        ctx.offsetA = batchOffsetInfo.alignA *
            (batchIndex % batchOffsetInfo.modA + batchIndex / batchOffsetInfo.divisorA);
        ctx.offsetB = batchOffsetInfo.alignB *
            (batchIndex % batchOffsetInfo.modB + batchIndex / batchOffsetInfo.divisorB);
        ctx.offsetBias = MATMUL_MODULE(BatchLoop)->GetBiasBatchSrcOffset();
    }
    
    __aicore__ inline void ComputeMultiIter(LocalTensor<SrcT>& a1, LocalTensor<SrcT>& b1, bool enPartialSum,
        BatchSchedulerContext& ctx)
    {
        // init split params for left and right matrix (k loop unrelated)
        BASE_MODULE::InitSplitAParams(ctx.aL0Params);
        BASE_MODULE::InitSplitBParams(ctx.bL0Params);
        // start k outer loop
        MATMUL_MODULE(KLoop)->OuterStart();
        do {
            // load bias in l1 and broadcast to cmatrix
            MATMUL_MODULE(BiasScheduler)->CopyIn(
                MATMUL_MODULE(NLoop)->GetBaseShape(), 1,
                ctx.offsetBias +
                (MATMUL_MODULE(BatchLoop)->GetBatchIndex() * MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN()) +
                MATMUL_MODULE(NLoop)->GetOuterIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
            // update k outer loop related params
            int32_t sL0CInit;
            int32_t sL0CLast;
            BASE_MODULE::UpdateSplitParams(enPartialSum, ctx.aL0Params, ctx.bL0Params, sL0CInit, sL0CLast);
            BASE_MODULE::MacroCompute(a1, b1, ctx, sL0CInit, sL0CLast);
        } while (MATMUL_MODULE(KLoop)->OuterNext());
    }

    __aicore__ inline void ComputeOneIter(LocalTensor<SrcT>& a1, LocalTensor<SrcT>& b1, bool enPartialSum,
        BatchSchedulerContext& ctx)
    {
        // load bias in l1 and broadcast to cmatrix
        MATMUL_MODULE(BiasScheduler)->CopyIn(
            MATMUL_MODULE(NLoop)->GetBaseShape(), 1,
            ctx.offsetBias +
            (MATMUL_MODULE(BatchLoop)->GetBatchIndex() * MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN()) +
            (MATMUL_MODULE(NLoop)->GetOuterIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN()));

        // init split params for left and right matrix (k loop unrelated)
        BASE_MODULE::InitSplitAParams(ctx.aL0Params);
        BASE_MODULE::InitSplitBParams(ctx.bL0Params);
        // start k outer loop
        MATMUL_MODULE(KLoop)->OuterStart();
        // update k outer loop related params
        BASE_MODULE::UpdateSplitParams(ctx.aL0Params, ctx.bL0Params);
        int32_t sL0CInit = enPartialSum ? 0 : 1;
        BASE_MODULE::MacroCompute(a1, b1, ctx, sL0CInit, 0);
    }

    __aicore__ inline void CopyOut(const GlobalTensor<DstT>& gm, const BatchSchedulerContext& ctx,
        int32_t enAtomic, bool enSequentialWrite)
    {
        event_t eventIDMToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
        BASE_MODULE::GetBatchResult(gm, ctx, enAtomic, enSequentialWrite);
        SetFlag<HardEvent::M_MTE1>(eventIDMToMte1);
        WaitFlag<HardEvent::M_MTE1>(eventIDMToMte1);
        if constexpr (ToMatmulConfig(MM_CFG).enableUBReuse && !ToMatmulConfig(MM_CFG).enableL1CacheUB) {
            event_t eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
        } else if constexpr (ToMatmulConfig(MM_CFG).enableL1CacheUB) {
            if ((MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetDepthAL1CacheUB() == 0 && A_TYPE::format == CubeFormat::ND) ||
                (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetDepthBL1CacheUB() == 0 && B_TYPE::format == CubeFormat::ND)) {
                event_t eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            }
        }
    }

    __aicore__ inline void CopyOut(const LocalTensor<DstT>& ubCmatrix, const BatchSchedulerContext& ctx,
        int32_t enAtomic, bool enSequentialWrite)
    {
        event_t eventIDMToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
        BASE_MODULE::GetBatchResult(ubCmatrix, ctx, enAtomic, enSequentialWrite);
        SetFlag<HardEvent::M_MTE1>(eventIDMToMte1);
        WaitFlag<HardEvent::M_MTE1>(eventIDMToMte1);
        event_t eventIDVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMte2);
    }
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_SCHEDULER_BATCH_BATCH_SCHEDULER_V200_H