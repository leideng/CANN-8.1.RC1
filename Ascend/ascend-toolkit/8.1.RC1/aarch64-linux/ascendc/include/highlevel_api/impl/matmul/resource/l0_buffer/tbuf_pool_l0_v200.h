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
 * \file tbuf_pool_l0_v200.h
 * \brief
 */
#ifndef IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_V200_H
#define IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_V200_H

#include "tbuf_pool_l0_intf.h"
#include "tbuf_pool_l0_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    TBufPoolL0 is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    TBufPoolL0 is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, typename A_TYPE, typename B_TYPE, const auto& MM_CFG>
class TBufPoolL0<IMPL, A_TYPE, B_TYPE, MM_CFG,
    enable_if_t<MatmulFeatureTrait<MM_CFG>::IsNeedUB() && (DoMatmulMDL(MM_CFG) ||
        isNormEnableScheduler<A_TYPE, MM_CFG> || IsBmmEnableScheduler<A_TYPE, MM_CFG> ||
        DoMatmulSpecialMDL(MM_CFG) || IsBasicBlockEnable<MM_CFG> || DoMatmulIBShareNorm(MM_CFG) ||
        IsIntrablock<MM_CFG>) && !IsA2B2Shared(MM_CFG) && !IsL0Cache<A_TYPE, MM_CFG>()>>
    : public TBufPoolL0Base<IMPL, A_TYPE, B_TYPE, MM_CFG>
{
public:
    using BASE_MODULE = AscendC::Impl::Detail::TBufPoolL0Base<IMPL, A_TYPE, B_TYPE, MM_CFG>;
    __aicore__ inline TBufPoolL0() {}
    __aicore__ inline ~TBufPoolL0() {
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1Ping_);
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1Pong_);
        GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdMToMte1Ping_);
        GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdMToMte1Pong_);
    };

    __aicore__ inline void Init()
    {
        BASE_MODULE::Init();
        eventIdMToMte1Ping_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
        eventIdMToMte1Pong_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1Ping_);
        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1Pong_);
    }

private:
    event_t eventIdMToMte1Ping_;
    event_t eventIdMToMte1Pong_;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_V200_H
