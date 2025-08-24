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
 * \file scheduler_special_mdl.h
 * \brief
 */
#ifndef IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_SPECIAL_MDL_H
#define IMPL_MATMUL_SCHEDULER_BASE_SCHEDULER_SPECIAL_MDL_H

#include "scheduler_intf.h"
#include "scheduler_mdl_base.h"

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
class MatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TR_MODE, enable_if_t<DoMatmulSpecialMDL(MM_CFG)>>
    : public MatmulMDLSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TR_MODE>
{
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
    MATMUL_USE_MODULE(LoadToA2);
    MATMUL_USE_MODULE(LoadToB2);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(MmadCompute);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(CopyCubeOut);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);

    using TransT = typename A_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using DstT = typename C_TYPE::T;

public:
    using BASE_MODULE =
        AscendC::Impl::Detail::MatmulMDLSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, TR_MODE>;

    __aicore__ inline bool ScheduleOnce(bool enPartialSum)
    {
        if (!BASE_MODULE::MoveNext()) {
            return false;
        }
        if (!enPartialSum) {
            MATMUL_MODULE(CubeOutBuffer)->AllocTensor();
        }
        MATMUL_MODULE(BiasScheduler)->SetBias(MATMUL_MODULE(BiasScheduler)->IsBias() && !enPartialSum);
        Compute(enPartialSum);
        return true;
    }

    __aicore__ void GetResult(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        static_assert(ToMatmulConfig(MM_CFG).scheduleType != ScheduleType::OUTER_PRODUCT, "Unsupported scheduleType");
        GetResultImpl(gm, enAtomic, enSequentialWrite);
    }

private:
    __aicore__ constexpr auto OneBlockSize() const {
        return MATMUL_MODULE(MLoop)->GetBaseBlockShape() * MATMUL_MODULE(NLoop)->GetBaseBlockShape() * CUBE_MAX_SIZE;
    }

    template <typename L1In, typename GMOut>
    __aicore__ void GetResultAlongNAxis(L1In& co1Local, GMOut& gm, bool enSequentialWrite = false)
    {
        auto loopModule = MATMUL_MODULE(NLoop);
        loopModule->InnerStart();
        do {
            auto offset = loopModule->GetInnerIdx() * OneBlockSize();
            if (enSequentialWrite) {
                MATMUL_MODULE(CopyCubeOut)
                    ->template Copy<true>(
                        gm, co1Local[offset], MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                        MATMUL_MODULE(MLoop)->GetBaseShape(), MATMUL_MODULE(NLoop)->GetBaseShape(),
                        MATMUL_MODULE(MLoop)->GetBaseBlockShape(), MATMUL_MODULE(NLoop)->GetBaseBlockShape());
            } else {
                MATMUL_MODULE(CopyCubeOut)
                    ->template Copy<false>(
                        gm, co1Local[offset], MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                        MATMUL_MODULE(MLoop)->GetBaseShape(), MATMUL_MODULE(NLoop)->GetBaseShape(),
                        MATMUL_MODULE(MLoop)->GetBaseBlockShape(), MATMUL_MODULE(NLoop)->GetBaseBlockShape());
            }
        } while (loopModule->InnerNext());
    }

    __aicore__ void GetResultImpl(const GlobalTensor<DstT>& gm, uint8_t enAtomic, bool enSequentialWrite)
    {
        if constexpr (C_TYPE::format != CubeFormat::ND && C_TYPE::format != CubeFormat::ND_ALIGN &&
            C_TYPE::format != CubeFormat::NZ) {
            ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR,
                "Data format of C matrix should be ND, ND_ALIGN or NZ."); });
        }
        // remove dependency conflicts only for scene which is not db
        auto co1Local = MATMUL_MODULE(CubeOutBuffer)->GetTensor();
        MATMUL_MODULE(CubeOutBuffer)->EnQue(co1Local);
        MATMUL_MODULE(CubeOutBuffer)->DeQue();
        BASE_MODULE::SetAtomic(enAtomic);
        GetResultAlongNAxis(co1Local, gm);
        BASE_MODULE::ClearAtomic(enAtomic);
        MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
    }

    __aicore__ void ComputeTilesAlongNAxis(int32_t kOuterIdx, bool enPartialSum = false)
    {
        bool isATranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
        bool isBTranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();

        auto loopModule = MATMUL_MODULE(NLoop);
        loopModule->InnerStart();

        auto aL1 = BASE_MODULE::InitSplitAParams();
        do {
            auto bL1 = BASE_MODULE::InitSplitBParams();

            MATMUL_MODULE(KLoop)->InnerStart();
            LocalTensor<TransT> a1;
            LocalTensor<TransT> b1;
            BASE_MODULE::CopyIn(a1, b1);
            bool sL0CInit = false;
            bool sL0CLast = false;
            BASE_MODULE::SplitPrepare(enPartialSum, isATranspose, isBTranspose, aL1, bL1, sL0CInit, sL0CLast);

            // Split bias
            SplitBias(bL1.axisL0Len);
            // prepare for Split
            int32_t kL1Stride = MATMUL_MODULE(KLoop)->GetBaseBlockShape() * BASE_MODULE::c0Size_;
            // start k inner loop
            MATMUL_MODULE(KLoop)->InnerStart();
            auto offset = loopModule->GetInnerIdx() * OneBlockSize();
            do {
                // allocate L0 buffer
                ComputeKDB(a1, b1, aL1, bL1, offset, isATranspose, isBTranspose, sL0CInit, sL0CLast);
                aL1.kAxisL1Offset += kL1Stride;
                bL1.kAxisL1Offset += kL1Stride;
            } while (MATMUL_MODULE(KLoop)->InnerNext());

            BASE_MODULE::DoPreloadAWait();
        } while (loopModule->InnerNext());
    }

    __aicore__ inline void SplitBias(const int32_t dataLen)
    {
        auto bias = MATMUL_MODULE(BiasScheduler)->CopyIn(MATMUL_MODULE(NLoop)->GetBaseShape(), 1,
            MATMUL_MODULE(NLoop)->GetInnerIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
        MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, dataLen);
        MATMUL_MODULE(BiasScheduler)->Free(bias);
    }

    __aicore__ inline void Compute(bool enPartialSum = false)
    {
        int32_t curKaOuterIdx = 0, curKbOuterIdx = 0;
        MATMUL_MODULE(KLoop)->OuterStart();
        do {
            ComputeTilesAlongNAxis(MATMUL_MODULE(KLoop)->GetOuterIdx(), enPartialSum);
            BASE_MODULE::ClearL1BufferCache(curKaOuterIdx, curKbOuterIdx);
        } while (MATMUL_MODULE(KLoop)->OuterNext());

        // clear L1 buffers
        BASE_MODULE::ResetCopyInBuffer();
    }

    __aicore__ inline void ComputeKDB(const LocalTensor<TransT>& a1, const LocalTensor<TransT>& b1,
        const SplitParams& aL0Params, const SplitParams& bL0Params, const int32_t offset,
        const bool isATranspose, const bool isBTranspose, const bool sL0CInit, const bool sL0CLast)
    {
        MATMUL_MODULE(TBufPoolL0)->Allocate();
        LocalTensor<TransT> a2 = BASE_MODULE::SplitA(a1, aL0Params, isATranspose);
        LocalTensor<TransT> b2 = BASE_MODULE::SplitB(b1, bL0Params, isBTranspose);
        MATMUL_MODULE(TBufPoolL0)->EnQue();
        MATMUL_MODULE(TBufPoolL0)->DeQue();
        BASE_MODULE::CubeCompute(MATMUL_MODULE(CubeOutBuffer)->GetTensor()[offset], a2, b2, aL0Params.axisL0Len,
            bL0Params.axisL0Len, MATMUL_MODULE(KLoop)->GetBaseShape(), isATranspose, isBTranspose, sL0CInit, sL0CLast);
        MATMUL_MODULE(TBufPoolL0)->Free();
        MATMUL_MODULE(BiasScheduler)->Free();
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC

#endif
