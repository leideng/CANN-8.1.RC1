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
* \file bias_scheduler_v200.h
* \brief
*/

#ifndef IMPL_MATMUL_SCHEDULER_BIAS_BIAS_SCHEDULER_V200_H
#define IMPL_MATMUL_SCHEDULER_BIAS_BIAS_SCHEDULER_V200_H

#include "bias_scheduler_intf.h"
#include "bias_scheduler_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {

/**
 * BiasScheduler: responsible for copy bias data management.
 * This module provides ablities to copy bias data in C2 or L0C.
 * We retain the freedom to make incompatible changes, but do not guarantee the stability.
 * BiasScheduler is only for internal usage, does not support extension or customized specialization!
 */
template <typename IMPL, class A_TYPE, class B_TYPE, class BIAS_TYPE, const auto &MM_CFG>
class BiasScheduler<IMPL, A_TYPE, B_TYPE, BIAS_TYPE, MM_CFG, enable_if_t<
    MatmulFeatureTrait<MM_CFG>::IsNeedUB() &&
    ToMatmulConfig(MM_CFG).enableSetBias &&
    (PhyPosIsUB(BIAS_TYPE::pos) || PhyPosIsGM(BIAS_TYPE::pos)) &&
    (DoMatmulMDL(MM_CFG) || isNormEnableScheduler<A_TYPE, MM_CFG> ||
    IsBmmEnableScheduler<A_TYPE, MM_CFG> || IsBasicBlockEnable<MM_CFG> || DoMatmulIBShareNorm(MM_CFG))>>
    : public BiasSchedulerBase<IMPL,  A_TYPE,  B_TYPE, BIAS_TYPE, MM_CFG>
{
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(CopyBiasIn);

    using BiasT = typename BIAS_TYPE::T;

public:
    using BASE_MODULE = AscendC::Impl::Detail::BiasSchedulerBase<IMPL, A_TYPE, B_TYPE, BIAS_TYPE, MM_CFG>;

    __aicore__ inline BiasScheduler() = default;
    __aicore__ inline ~BiasScheduler() = default;
    __aicore__ inline void Init(int32_t batchNum = 0) {}

    __aicore__ inline LocalTensor<BiasT> CopyIn(int32_t dataLen, int32_t dataNum = 1, int32_t srcOffset = 0)
    {
        LocalTensor<BiasT> biasC1;
        if (BASE_MODULE::enableBias_ && MATMUL_MODULE(KLoop)->GetOuterIdx() == 0) {
            MATMUL_MODULE(CopyBiasIn)->Copy(biasC1, BASE_MODULE::srcTensor_, dataLen, dataNum,
                                            srcOffset + BASE_MODULE::singleOffset_);
        }
        return biasC1;
    }

    __aicore__ inline void Free(LocalTensor<BiasT> &biasC1) {}

    __aicore__ inline void SplitLoad(LocalTensor<BiasT> &biasC1, int32_t dataLen = 0, int32_t srcOffset = 0) {}

    __aicore__ inline void Free() {}

    __aicore__ inline void End() {}
};

}  // namespace Detail
}  // namespace Impl
}  // namespace Gemm
#endif // _BIAS_SCHEDULER_V200_H_
