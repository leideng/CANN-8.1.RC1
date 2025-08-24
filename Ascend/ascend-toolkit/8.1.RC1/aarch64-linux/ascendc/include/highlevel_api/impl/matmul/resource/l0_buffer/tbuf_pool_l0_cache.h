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
 * \file tbuf_pool_l0_cache.h
 * \brief
 */
#ifndef IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_CACHE_H
#define IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_CACHE_H

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
    enable_if_t<!MatmulFeatureTrait<MM_CFG>::IsNeedUB() && (DoMatmulMDL(MM_CFG) ||
    isNormEnableScheduler<A_TYPE, MM_CFG>) && IsL0Cache<A_TYPE, MM_CFG>()>>
    : public TBufPoolL0Base<IMPL, A_TYPE, B_TYPE, MM_CFG>
{
public:
    using BASE_MODULE = AscendC::Impl::Detail::TBufPoolL0Base<IMPL, A_TYPE, B_TYPE, MM_CFG>;
    __aicore__ inline TBufPoolL0() = default;
    __aicore__ inline ~TBufPoolL0() = default;

    template <TPosition Pos, typename T>
    __aicore__ inline LocalTensor<T> GetBuffer()
    {
        LocalTensor<T> tempTensor;
        if constexpr (Pos == TPosition::A2) {
            tempTensor = BASE_MODULE::l0aBuf_.template Get<T>();
            if ((l0aPingPongFlag_ & 0x1) != 0) {
                if constexpr (IsSameType<T, int4b_t>::value) {
                    tempTensor = tempTensor[L0AUF_SIZE / sizeof(T)];
                } else {
                    tempTensor = tempTensor[L0AUF_SIZE / 2 / sizeof(T)];
                }
            }
        } else {
            tempTensor = BASE_MODULE::l0bBuf_.template Get<T>();
            if ((l0bPingPongFlag_ & 0x1) != 0) {
                if constexpr (IsSameType<T, int4b_t>::value) {
                    tempTensor = tempTensor[L0BUF_SIZE / sizeof(T)];
                } else {
                    tempTensor = tempTensor[L0BUF_SIZE / 2 / sizeof(T)];
                }
            }
        }
        return tempTensor;
    }

    template <TPosition Pos>
    __aicore__ inline bool Hit(uint32_t pos = 0) {
        bool hit = false;
        if constexpr (Pos == TPosition::A2) {
            bool hitPingCache = pos == l0aPingCachedPos_;
            bool hitPongCache = pos == l0aPongCachedPos_;
            if (hitPingCache) {
                hit = true;
                l0aPingPongFlag_ = 0;
            } else if (hitPongCache) {
                hit = true;
                l0aPingPongFlag_ = 1;
            } else {
                hit = false;
                l0aPingPongFlag_ = 0; // singleMK == baseMK, no need to pingpong
                if ((l0aPingPongFlag_ & 0x1) == 0) {
                    l0aPingCachedPos_ = pos;
                } else {
                    l0aPongCachedPos_ = pos;
                }
                ++l0aLoadTimes_;
            }
        } else {
            bool hitPingCache = pos == l0bPingCachedPos_;
            bool hitPongCache = pos == l0bPongCachedPos_;
            if (hitPingCache) {
                hit = true;
                l0bPingPongFlag_ = 0;
            } else if (hitPongCache) {
                hit = true;
                l0bPingPongFlag_ = 1;
            } else {
                hit = false;
                l0bPingPongFlag_ = BASE_MODULE::l0PingPongFlag_; // l0b pingpong is same as l0PingPongFlag_
                if ((l0bPingPongFlag_ & 0x1) == 0) {
                    l0bPingCachedPos_ = pos;
                } else {
                    l0bPongCachedPos_ = pos;
                }
                ++l0bLoadTimes_;
            }
        }
        return hit;
    }

    __aicore__ inline void ResetCache() {
        WaitFlag<HardEvent::M_MTE1>(0); // To solve sync problem between muti objects, need to wait pingpong here
        SetFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(1);
        SetFlag<HardEvent::M_MTE1>(1);
        l0aPingPongFlag_ = 1;
        l0bPingPongFlag_ = 1;
        l0aPingCachedPos_ = -1;
        l0aPongCachedPos_ = -1;
        l0bPingCachedPos_ = -1;
        l0bPongCachedPos_ = -1;
        l0aLoadTimes_ = 0;
        l0bLoadTimes_ = 0;
    }

private:
    uint16_t l0aPingPongFlag_{1};
    uint16_t l0bPingPongFlag_{1};
    int32_t l0aPingCachedPos_{-1};
    int32_t l0aPongCachedPos_{-1};
    int32_t l0bPingCachedPos_{-1};
    int32_t l0bPongCachedPos_{-1};
    int32_t l0aLoadTimes_{0};
    int32_t l0bLoadTimes_{0};

};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_CACHE_H
