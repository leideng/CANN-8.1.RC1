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
 * \file n_loop_norm.h
 * \brief
 */


#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_NORM_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_NORM_H

#include "n_loop_intf.h"
#include "n_loop_norm_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    NLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    NLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, const auto& MM_CFG>
class NLoop<IMPL, A_TYPE, MM_CFG, enable_if_t<(isNormEnableScheduler<A_TYPE, MM_CFG> || DoMatmulIBShareNorm(MM_CFG))
    && !MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()>> : public NLoopNormBase<IMPL, A_TYPE, MM_CFG>
{
    MATMUL_USE_MODULE(MatmulShapeTiling);
public:
    using BASE_MODULE = AscendC::Impl::Detail::NLoopNormBase<IMPL, A_TYPE, MM_CFG>;
    __aicore__ inline NLoop() = default;
    __aicore__ inline ~NLoop() = default;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _N_LOOP_NORM_H_
