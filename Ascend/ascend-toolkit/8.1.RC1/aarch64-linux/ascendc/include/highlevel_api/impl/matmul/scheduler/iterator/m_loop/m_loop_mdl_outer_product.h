/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file m_loop_mdl_outer_product.h
 * \brief
 */


#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_M_LOOP_M_LOOP_MDL_OUTER_PRODUCT_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_M_LOOP_M_LOOP_MDL_OUTER_PRODUCT_H

#include "m_loop_intf.h"
#include "m_loop_mdl_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    MLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, const auto& MM_CFG>
class MLoop<IMPL, A_TYPE, MM_CFG, enable_if_t<DoMatmulMDL(MM_CFG) &&
    MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()>>
    : public MLoopMDLBase<IMPL, A_TYPE, MM_CFG>
{
    MATMUL_USE_MODULE(MatmulShapeTiling);
public:
    using BASE_MODULE = AscendC::Impl::Detail::MLoopMDLBase<IMPL, A_TYPE, MM_CFG>;
    __aicore__ inline MLoop() = default;
    __aicore__ inline ~MLoop() = default;

    __aicore__ inline bool InnerNext()
    {
        if constexpr (IsBasicM(MM_CFG)) {
            return false;
        } else {
            if (IsL0DoubleBuffer()) {
                BASE_MODULE::innerIndex_ += DB_FACTOR;
            } else {
                ++BASE_MODULE::innerIndex_;
            }
            BASE_MODULE::UpdateInnerParams();
            CalcDBLoopNum();
            return !BASE_MODULE::InnerEnd();
        }
    }

    __aicore__ inline void InnerStart()
    {
        BASE_MODULE::innerIndex_ = BASE_MODULE::innerStartIdx_;
        CalcDBLoopNum();
        BASE_MODULE::UpdateInnerParams();
    }

    __aicore__ inline bool IsL0DoubleBuffer()
    {
        if constexpr(ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_M) {
            return false; // ORDER_M do not support m axis l0 double buffer
        } else {
            if (BASE_MODULE::outerIndex_ + 1 < BASE_MODULE::outerIter_) {
                return BASE_MODULE::innerIndex_ + DB_FACTOR <= BASE_MODULE::innerStartIdx_ + BASE_MODULE::innerIter_;
            } else {
                return (BASE_MODULE::innerIndex_ + DB_FACTOR < BASE_MODULE::totalIter_) ||
                    (BASE_MODULE::innerIndex_ + DB_FACTOR == BASE_MODULE::totalIter_ &&
                    BASE_MODULE::tailBaseShape_ == MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM());
            }
        }
    }

    __aicore__ inline uint32_t GetL0DBLoopNum() const
    {
        return l0dbLoopNum;
    }

private:
    __aicore__ inline void CalcDBLoopNum()
    {
        l0dbLoopNum = IsL0DoubleBuffer() ? DB_FACTOR : 1;
    }

    // DBLoop
    uint32_t l0dbLoopNum = 1;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _M_LOOP_MDL_OUTER_PRODUCT_H_