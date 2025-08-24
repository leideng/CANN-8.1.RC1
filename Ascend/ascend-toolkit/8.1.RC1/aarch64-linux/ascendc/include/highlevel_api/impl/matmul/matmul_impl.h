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
 * \file matmul_impl.h
 * \brief
 */
#ifndef IMPL_MATMUL_MATMUL_IMPL_H
#define IMPL_MATMUL_MATMUL_IMPL_H

#include "matmul_impl_base.h"

namespace AscendC {

// Match Policy with CallBack paramter
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
class MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY, enable_if_t<A_TYPE::layout == LayoutMode::NONE>>
    : public MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>
{
private:
    using SrcT = typename A_TYPE::T;
    using DstT = typename C_TYPE::T;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    MATMUL_USE_MODULE(Scheduler);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulAntiQuantProcessor);
    MATMUL_USE_MODULE(MatmulSubBlockInfo);
public:
    using BASE_MODULE = MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    __aicore__ inline MatmulImpl() {}

    __aicore__ inline void SetAntiQuantScalar(const SrcT offsetScalar, const SrcT scaleScalar)
    {
        MATMUL_MODULE(MatmulAntiQuantProcessor)->SetAntiQuantScalar(offsetScalar, scaleScalar);
    }

    __aicore__ inline void SetAntiQuantVector(const LocalTensor<SrcT> &offsetTensor,
        const LocalTensor<SrcT> &scaleTensor)
    {
        MATMUL_MODULE(MatmulAntiQuantProcessor)->SetAntiQuantVector(offsetTensor, scaleTensor);
    }

#if __CCE_AICORE__ < 220
    // v100, v200
    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0,
        bool enSequentialWrite = false, bool waitIterateAll = false, bool fakeMsg = false)
    {
#if __CCE_AICORE__ == 200
        GlobalTensor<uint64_t> global;
        global.SetGlobalBuffer((__gm__ uint64_t*)0);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE>(global);
#endif
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        while (BASE_MODULE::Iterate()) {
            BASE_MODULE::GetTensorC(gm, enAtomic);
            if constexpr (ToMatmulConfig(MM_CFG).enableUBReuse && !ToMatmulConfig(MM_CFG).enableL1CacheUB) {
                event_t eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            } else if constexpr (ToMatmulConfig(MM_CFG).enableL1CacheUB) {
                if ((tiling.GetDepthAL1CacheUB() == 0 && A_TYPE::format == CubeFormat::ND) ||
                    (tiling.GetDepthBL1CacheUB() == 0 && B_TYPE::format == CubeFormat::ND)) {
                    event_t eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                    SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
                }
            }
        }
    }

    // v100, v200
    template <bool sync = true>
    __aicore__ inline void IterateAll(const LocalTensor<DstT>& ubCmatrix, uint8_t enAtomic = 0)
    {
#if __CCE_AICORE__ == 200
        GlobalTensor<uint64_t> global;
        global.SetGlobalBuffer((__gm__ uint64_t*)0);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE>(global);
#endif
        (void)(enAtomic);
        while (BASE_MODULE::Iterate()) {
            BASE_MODULE::GetTensorC(ubCmatrix);
            event_t eventIDVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIDVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIDVToMte2);
        }
    }
#else
    // v220
    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0,
        bool enSequentialWrite = false, bool waitIterateAll = false, bool fakeMsg = false)
    {
        if constexpr (ToMatmulConfig(MM_CFG).intraBlockPartSum) {
            MATMUL_MODULE(MatmulSubBlockInfo)->SetFakeMsg(fakeMsg);
            MATMUL_MODULE(Scheduler)->Schedule(gm, enAtomic, enSequentialWrite, fakeMsg);
        } else {
            while (BASE_MODULE::Iterate()) {
                BASE_MODULE::GetTensorC(gm, enAtomic);
            }
        }
    }

    // v220
    template <bool sync = true>
    __aicore__ inline void IterateAll(const LocalTensor<DstT>& ubCmatrix, uint8_t enAtomic = 0)
    {
        while (BASE_MODULE::Iterate()) {
            BASE_MODULE::GetTensorC(ubCmatrix, enAtomic);
        }
    }
#endif
};

} // namespace AscendC

#endif
