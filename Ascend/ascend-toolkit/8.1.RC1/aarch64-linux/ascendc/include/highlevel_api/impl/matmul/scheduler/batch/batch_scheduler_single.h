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
 * \file batch_scheduler_single.h
 * \brief
 */
#ifndef IMPL_MATMUL_SCHEDULER_BATCH_BATCH_SCHEDULER_SINGLE_H
#define IMPL_MATMUL_SCHEDULER_BATCH_BATCH_SCHEDULER_SINGLE_H

#include "batch_scheduler_intf.h"

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
    !MatmulFeatureTrait<MM_CFG>::IsSupportCmatrixInitVal() &&
    (A_TYPE::layout == LayoutMode::NORMAL && ToMatmulConfig(MM_CFG).batchMode == BatchMode::SINGLE_LARGE_THAN_L1)>>
{
    MATMUL_USE_MODULE(BatchLoop);
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(MatmulQuantProcessor);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulTensorInfoA);
    MATMUL_USE_MODULE(MatmulTensorInfoB);
    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(Scheduler);
    MATMUL_USE_MODULE(BiasScheduler);

    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    using SrcT = typename A_TYPE::T;
    using DstT = typename C_TYPE::T;

public:
    __aicore__ inline BatchScheduler() = default;
    __aicore__ inline ~BatchScheduler() = default;

    __aicore__ inline void Init(const TCubeTiling *__restrict cubeTiling, TPipe *tpipe)
    {
        MATMUL_MODULE(Scheduler)->Init(cubeTiling, tpipe);
    }

    __aicore__ inline void End()
    {
        MATMUL_MODULE(CopyCubeInA)->Destroy();
        MATMUL_MODULE(CopyCubeInB)->Destroy();
        if constexpr (MatmulFeatureTrait<MM_CFG>::IsNeedUB()) {
            MATMUL_MODULE(CubeOutBuffer)->Destroy();
            MATMUL_MODULE(MatmulQuantProcessor)->Destroy();
        } else {
            MATMUL_MODULE(BiasScheduler)->End();
            MATMUL_MODULE(CubeOutBuffer)->Destroy();
            if constexpr (((IsSameTypeV<SrcT, int8_t> || IsSameTypeV<SrcT, int4b_t>) && IsSameTypeV<DstT, half>) ||
                (IsSameTypeV<SrcT, int8_t> && (IsSameTypeV<DstT, int8_t> || IsSameTypeV<DstT, uint8_t>))) {
                MATMUL_MODULE(MatmulQuantProcessor)->Destroy();
            }
        }
    }

    __aicore__ inline void Schedule(const LocalTensor<DstT>& dst, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite,
        const uint32_t matrixStrideA, const uint32_t matrixStrideB, const uint32_t matrixStrideC)
    {
        if constexpr (MatmulFeatureTrait<MM_CFG>::IsNeedUB()) {
            ScheduleImpl(dst, enPartialSum, enAtomic, enSequentialWrite, matrixStrideA, matrixStrideB, matrixStrideC);
        }
    }

    __aicore__ inline void Schedule(const GlobalTensor<DstT>& dst, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite,
        const uint32_t matrixStrideA, const uint32_t matrixStrideB, const uint32_t matrixStrideC)
    {
        ScheduleImpl(dst, enPartialSum, enAtomic, enSequentialWrite, matrixStrideA, matrixStrideB, matrixStrideC);
    }

private:
    template <class T>
    __aicore__ inline void ScheduleImpl(const T& dst, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite,
        const uint32_t matrixStrideA, const uint32_t matrixStrideB, const uint32_t matrixStrideC)
    {
        GlobalTensor<SrcT> aGlobal = MATMUL_MODULE(MatmulTensorInfoA)->GetGlobalTensor();
        GlobalTensor<SrcT> bGlobal = MATMUL_MODULE(MatmulTensorInfoB)->GetGlobalTensor();
        const auto batchLoop = MATMUL_MODULE(BatchLoop);
        const auto matmulShapeInfo = MATMUL_MODULE(MatmulShapeInfo);
        int32_t batchASize = matmulShapeInfo->GetSingleCoreM() * matmulShapeInfo->GetSingleCoreK();
        int32_t batchBSize = matmulShapeInfo->GetSingleCoreK() * matmulShapeInfo->GetSingleCoreN();
        for (batchLoop->OuterStart(); !batchLoop->OuterEnd(); batchLoop->OuterNext()) {
            MATMUL_MODULE(CopyCubeInA)->SetInput(aGlobal[batchLoop->GetBatchAIndex() * batchASize],
                matmulShapeInfo->IsTransposeA());
            MATMUL_MODULE(CopyCubeInB)->SetInput(bGlobal[batchLoop->GetBatchBIndex() * batchBSize],
                matmulShapeInfo->IsTransposeB());
            if constexpr (IsSameTypeV<SrcT, int8_t> && IsSameTypeV<DstT, half>) {
                if (MATMUL_MODULE(MatmulQuantProcessor)->GetMatmulQuantMode() == QuantMode_t::VDEQF16 &&
                    batchLoop->GetOuterIndex() > 0) { 
                    MATMUL_MODULE(MatmulQuantProcessor)->UpdateQuantTensor(matmulShapeInfo->GetSingleCoreN());
                }
            }
            MATMUL_MODULE(BiasScheduler)->SetSingleOffset(MATMUL_MODULE(BatchLoop)->GetBiasInputOffset());

            isFirstIter_ = true;
            while (MoveNext()) {
                MATMUL_MODULE(CubeOutBuffer)->AllocTensor();
                MATMUL_MODULE(Scheduler)->Compute(enPartialSum);
                CopyOut(dst, enAtomic, enSequentialWrite);
            }
            End();
        }
    }

    __aicore__ inline bool MoveNext()
    {
        if (unlikely(isFirstIter_)) {
            return MoveOnFirstIterate();
        } else {
            if constexpr (ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::UNDEF) {
                if (likely(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetIterateOrder() ==
                    static_cast<int>(IterateOrder::ORDER_M))) {
                    return MoveOnIterateOrderM();
                } else {
                    ASCENDC_ASSERT((MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetIterateOrder() ==
                        static_cast<int>(IterateOrder::ORDER_N)), {KERNEL_LOG(KERNEL_ERROR, "iterateOrder is %d , which should be ORDER_N",
                        MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetIterateOrder());
                    });
                    return MoveOnIterateOrderN();
                }
            } else if constexpr (ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_M) {
                return MoveOnIterateOrderM();
            } else {
                return MoveOnIterateOrderN();
            }
        }
        return true;
    }

    __aicore__ inline bool MoveOnFirstIterate()
    {
        isFirstIter_ = false;
        MATMUL_MODULE(MLoop)->OuterStart();
        MATMUL_MODULE(NLoop)->OuterStart();
        return true;
    }

    __aicore__ inline bool MoveOnIterateOrderM()
    {
        MATMUL_MODULE(CopyCubeInA)->Reset();
        if (!MATMUL_MODULE(MLoop)->OuterNext()) {
            MATMUL_MODULE(CopyCubeInB)->Reset();
            if (!MATMUL_MODULE(NLoop)->OuterNext()) {
                return false;
            }
            MATMUL_MODULE(MLoop)->OuterStart();
        }
        return true;
    }

    __aicore__ inline bool MoveOnIterateOrderN()
    {
        MATMUL_MODULE(CopyCubeInB)->Reset();
        if (!MATMUL_MODULE(NLoop)->OuterNext()) {
            MATMUL_MODULE(CopyCubeInA)->Reset();
            if (!MATMUL_MODULE(MLoop)->OuterNext()) {
                return false;
            }
            MATMUL_MODULE(NLoop)->OuterStart();
        }
        return true;
    }

    __aicore__ inline void CopyOut(const GlobalTensor<DstT>& gm, int32_t enAtomic, bool enSequentialWrite)
    {
        MATMUL_MODULE(Scheduler)->GetResult(gm[MATMUL_MODULE(BatchLoop)->GetOuterIndex() *
            MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM() * MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN()],
            enAtomic, enSequentialWrite);
        if constexpr (!MatmulFeatureTrait<MM_CFG>::IsNeedUB()) {
            return;
        }

        if constexpr (ToMatmulConfig(MM_CFG).enableUBReuse && !ToMatmulConfig(MM_CFG).enableL1CacheUB) {
            event_t eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
        } else if constexpr (ToMatmulConfig(MM_CFG).enableL1CacheUB) {
            if ((MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetDepthAL1CacheUB() == 0 &&
                A_TYPE::format == CubeFormat::ND) ||
                (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetDepthBL1CacheUB() == 0 &&
                B_TYPE::format == CubeFormat::ND)) {
                event_t eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            }
        }
    }

    __aicore__ inline void CopyOut(const LocalTensor<DstT>& ubCmatrix, int32_t enAtomic, bool enSequentialWrite)
    {
        MATMUL_MODULE(Scheduler)->GetResult(ubCmatrix[MATMUL_MODULE(BatchLoop)->GetOuterIndex() *
            MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM() * MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN()],
            enAtomic, enSequentialWrite);
        if constexpr (!MatmulFeatureTrait<MM_CFG>::IsNeedUB()) {
            return;
        }

        event_t eventIDVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMte2);
    }
private:
    bool isFirstIter_;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_SCHEDULER_BATCH_BATCH_SCHEDULER_SINGLE_H
