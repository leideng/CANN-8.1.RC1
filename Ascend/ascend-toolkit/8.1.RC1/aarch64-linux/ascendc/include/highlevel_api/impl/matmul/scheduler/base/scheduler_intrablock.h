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
 * \file scheduler_intrablock.h
 * \brief
 */
#ifndef IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_INTRABLOCK_H
#define IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_INTRABLOCK_H

#include "scheduler_norm.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <const auto& MM_CFG>
__aicore__ inline constexpr auto GetNormCFG()
{
    using CFG_TYPE = typename std::remove_cv<typename std::remove_reference<decltype(MM_CFG)>::type>::type;
    CFG_TYPE cfg = MM_CFG;
    if constexpr (IsSameTypeV<CFG_TYPE, MatmulApiStaticTiling>) {
        cfg.cfg.intraBlockPartSum = false;
    } else {
        cfg.intraBlockPartSum = false;
    }
    return cfg;
}

template <const auto& MM_CFG>
static constexpr auto CFG = GetNormCFG<MM_CFG>();
/*
    MatmulScheduler is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MatmulScheduler is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
    TriangularMode TR_MODE>
class MatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TR_MODE, enable_if_t<IsIntrablock<MM_CFG>>>
    : public MatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG<MM_CFG>, TR_MODE>
{
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(CopyCubeOut);
    MATMUL_USE_MODULE(LoadToA2);
    MATMUL_USE_MODULE(LoadToB2);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(MmadCompute);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(MatmulUnitFlag);

    using TransT = typename A_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using DstT = typename C_TYPE::T;

public:
    using BASE_MODULE =
        AscendC::Impl::Detail::MatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG<MM_CFG>, TR_MODE>;

    __aicore__ inline bool ScheduleOnce(bool enPartialSum)
    {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "IntraBlockPartSum does not support Iterate."); });
    }

    __aicore__ inline void Schedule(const GlobalTensor<DstT>& gm,
        uint8_t enAtomic = 0, bool enSequentialWrite = false, bool fakeMsg = false)
    {
#if __CCE_AICORE__ == 220
        if (fakeMsg) {
            while (BASE_MODULE::ScheduleOnce(false)) {
                BASE_MODULE::GetResult(gm, enAtomic, enSequentialWrite);
            }
        } else {
            ScheduleIntraBlock(gm, enAtomic, enSequentialWrite);
        }
#endif
    }

    __aicore__ inline void Reset() {}

private:
    __aicore__ inline void ScheduleIntraBlock(const GlobalTensor<DstT>& gm, uint8_t enAtomic, bool enSequentialWrite)
    {
        PreSplitB<false>(0);
        PreSplitB<true>(1);
        MATMUL_MODULE(NLoop)->OuterStart();
        do {
            MATMUL_MODULE(MLoop)->template OuterStart<true>();
            do {
                MATMUL_MODULE(NLoop)->InnerStart();
                do {
                    // alloc cube out tensor
                    MATMUL_MODULE(CubeOutBuffer)->AllocTensor();
                    Compute<false>();
                    BASE_MODULE::End();
                    // v1 does not support bias
                    MATMUL_MODULE(BiasScheduler)->SetBias(false);
                    Compute<true>();
                    MATMUL_MODULE(CopyCubeInA)->Reset();
                    GetResultImpl(gm, enAtomic, enSequentialWrite);
                } while (MATMUL_MODULE(NLoop)->InnerNext());
                MATMUL_MODULE(CopyCubeInA)->Reset();
            } while (MATMUL_MODULE(MLoop)->template OuterNext<true>());
        } while (MATMUL_MODULE(NLoop)->OuterNext());
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void PreSplitB(uint8_t subBlockIdx)
    {
        auto singleCoreK = MATMUL_MODULE(MatmulShapeInfo)->template GetSingleCoreK<IS_INTRA_BLOCK>();
        auto singleCoreN = MATMUL_MODULE(MatmulShapeInfo)->template GetSingleCoreN<IS_INTRA_BLOCK>();
        auto b1 = MATMUL_MODULE(CopyCubeInB)->AsyncLoadData(0, 0, singleCoreK, singleCoreN);
        bool isBTranspose = MATMUL_MODULE(MatmulShapeInfo)->template IsTransposeB<IS_INTRA_BLOCK>();
        auto kAxisL1Len = CeilAlign(singleCoreK, BLOCK_CUBE);
        MATMUL_MODULE(LoadToB2)->Prepare(isBTranspose, kAxisL1Len);
        MATMUL_MODULE(CopyCubeInB)->AwaitLoadData();
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(1);
        auto baseN = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        auto baseK = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        auto l0bStride = baseN * baseK;
        MATMUL_MODULE(NLoop)->template OuterStart<true, IS_INTRA_BLOCK>();
        int16_t axisL1Offset = 0;
        do {
            auto baseUseN = MATMUL_MODULE(NLoop)->template GetBaseShape<IS_INTRA_BLOCK>();
            MATMUL_MODULE(KLoop)->template OuterStart<IS_INTRA_BLOCK>();
            uint16_t offset = 0;
            int16_t kAxisL1Offset = 0;
            do {
                auto& bufferPool = MATMUL_MODULE(TBufPoolL0)->template Allocate<true>();
                LocalTensor<TransT> b2 = bufferPool.template GetBuffer<TPosition::B2, TransT, true>(subBlockIdx);
                b2 = b2[offset];
                MATMUL_MODULE(LoadToB2)->Load(b2, b1, singleCoreN, kAxisL1Len, baseUseN,
                    MATMUL_MODULE(KLoop)->template GetBaseShape<IS_INTRA_BLOCK>(),
                    axisL1Offset, kAxisL1Offset, isBTranspose);
                offset += l0bStride;
                axisL1Offset += baseN;
                kAxisL1Offset += baseK;
            } while (MATMUL_MODULE(KLoop)->template OuterEnd<IS_INTRA_BLOCK>());
        } while (MATMUL_MODULE(NLoop)->template OuterEnd<false, IS_INTRA_BLOCK>());
        MATMUL_MODULE(CopyCubeInB)->ClearLoadData(b1);
        SetFlag<HardEvent::MTE1_M>(3);
        WaitFlag<HardEvent::MTE1_M>(3);
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_MTE1>(1);
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void Compute()
    {
        auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        // init split params for left and right matrix
        SplitParams aL0Params = BASE_MODULE::template InitSplitAParams<IS_INTRA_BLOCK>();
        SplitParams bL0Params = BASE_MODULE::template InitSplitBParams<IS_INTRA_BLOCK>();

        // start K outer loop
        MATMUL_MODULE(KLoop)->template OuterStart<IS_INTRA_BLOCK>();
        do {
            int32_t kOuterIdx = MATMUL_MODULE(KLoop)->template GetOuterIdx<IS_INTRA_BLOCK>();
            // CopyIn
            LocalTensor<TransT> a1 = MATMUL_MODULE(CopyCubeInA)->LoadData(
                MATMUL_MODULE(MLoop)->template GetOuterIdx<true>(),
                kOuterIdx, MATMUL_MODULE(MLoop)->template GetTileShape<IS_INTRA_BLOCK>(),
                MATMUL_MODULE(KLoop)->template GetTileShapeA<IS_INTRA_BLOCK>());
            if constexpr (!IS_INTRA_BLOCK) {
                // only v0 core support bias
                LocalTensor<BiasT> bias = MATMUL_MODULE(BiasScheduler)->CopyIn(
                    MATMUL_MODULE(NLoop)->template GetBaseShape<false>(), 1,
                    MATMUL_MODULE(NLoop)->GetInnerIdx() * tiling.GetBaseN());
                MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, bL0Params.axisL0Len);
            }
            // update some params in SplitParams which is related to k loop
            BASE_MODULE::template UpdateSplitParams<IS_INTRA_BLOCK>(aL0Params, bL0Params);
            int32_t sL0CInit = 0;
            int32_t sL0CLast = 0;
            if constexpr (!IS_INTRA_BLOCK) {
                if (unlikely(kOuterIdx == 0)) {
                    sL0CInit = 1;
                }
            } else {
                if constexpr (EnUnitFlag(MM_CFG)) {
                    sL0CLast = (kOuterIdx == MATMUL_MODULE(KLoop)->template GetTotalIter<true>() - 1 ) ? 1 : 0;
                }
            }

            bool isATranspose = MATMUL_MODULE(MatmulShapeInfo)->template IsTransposeA<IS_INTRA_BLOCK>();
            MATMUL_MODULE(LoadToA2)->Prepare(isATranspose, aL0Params.kAxisL1Len, aL0Params.axisL1Len);
            int32_t kL0Len = MATMUL_MODULE(KLoop)->template GetTileShapeA<IS_INTRA_BLOCK>();
            // allocate L0 buffer
            // Split
            auto& bufferPool = MATMUL_MODULE(TBufPoolL0)->Allocate();
            LocalTensor<TransT> a2 = bufferPool.template GetBuffer<TPosition::A2, TransT>();
            MATMUL_MODULE(LoadToA2)->Load(a2, a1, aL0Params.axisL1Len, aL0Params.kAxisL1Len,
                aL0Params.axisL0Len, kL0Len, aL0Params.axisL1Offset, aL0Params.kAxisL1Offset, isATranspose);
            auto kTotalIter = MATMUL_MODULE(KLoop)->template GetTotalIter<IS_INTRA_BLOCK>();
            int32_t posB =
                (kOuterIdx + MATMUL_MODULE(NLoop)->GetInnerIdx() * kTotalIter) % (tiling.GetStepN() * kTotalIter);
            LocalTensor<TransT> b2 =
                bufferPool.template GetBuffer<TPosition::B2, TransT, IS_INTRA_BLOCK>((uint8_t)IS_INTRA_BLOCK);
            b2 = b2[posB * tiling.GetBaseN() * tiling.GetBaseK()];
            bufferPool.EnQue();
            bufferPool.DeQue();
            // prepare params and compute
            uint8_t unitFlag = 0;
            if constexpr (IS_INTRA_BLOCK) {
                unitFlag = MATMUL_MODULE(MatmulUnitFlag)->GetUnitFlag(sL0CLast);
            }
            bool isBias;
            bool cmatrixSource;
            bool cmatrixInitVal;
            BASE_MODULE::UpdateBiasParams(IS_INTRA_BLOCK, sL0CInit, cmatrixSource, cmatrixInitVal, isBias);
            MATMUL_MODULE(MmadCompute)->Compute(MATMUL_MODULE(CubeOutBuffer)->GetTensor(), a2, b2,
                aL0Params.axisL0Len, kL0Len, bL0Params.axisL0Len, isATranspose,
                MATMUL_MODULE(MatmulShapeInfo)->template IsTransposeB<IS_INTRA_BLOCK>(),
                unitFlag, cmatrixSource, cmatrixInitVal, isBias);
            bufferPool.Free();
            if constexpr (!IS_INTRA_BLOCK) {
                MATMUL_MODULE(BiasScheduler)->Free();
            }

            MATMUL_MODULE(CopyCubeInA)->ClearLoadData(
                a1, MATMUL_MODULE(MLoop)->template GetOuterIdx<true>(), kOuterIdx);
        } while (MATMUL_MODULE(KLoop)->template OuterNext<IS_INTRA_BLOCK>());
    }

    __aicore__ inline void GetResultImpl(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0,
        bool enSequentialWrite = false)
    {
        auto co1Local = MATMUL_MODULE(CubeOutBuffer)->GetTensor();
        MATMUL_MODULE(CubeOutBuffer)->EnQue(co1Local);
        MATMUL_MODULE(CubeOutBuffer)->DeQue();
        BASE_MODULE::SetAtomic(enAtomic);

        if (enSequentialWrite) {
            MATMUL_MODULE(CopyCubeOut)->template Copy<true>(gm, co1Local,
                MATMUL_MODULE(MLoop)->template GetOuterIdx<true>(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                MATMUL_MODULE(MLoop)->template GetBaseShape<true>(),
                MATMUL_MODULE(NLoop)->template GetBaseShape<true>(),
                MATMUL_MODULE(MLoop)->template GetBaseBlockShape<true>(),
                MATMUL_MODULE(NLoop)->template GetBaseBlockShape<true>());
        } else {
            MATMUL_MODULE(CopyCubeOut)->template Copy<false>(gm, co1Local,
                MATMUL_MODULE(MLoop)->template GetOuterIdx<true>(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                MATMUL_MODULE(MLoop)->template GetBaseShape<true>(),
                MATMUL_MODULE(NLoop)->template GetBaseShape<true>(),
                MATMUL_MODULE(MLoop)->template GetBaseBlockShape<true>(),
                MATMUL_MODULE(NLoop)->template GetBaseBlockShape<true>());
        }

        BASE_MODULE::ClearAtomic(enAtomic);
        MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
        MATMUL_MODULE(CubeOutBuffer)->Destroy();
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC

#endif
