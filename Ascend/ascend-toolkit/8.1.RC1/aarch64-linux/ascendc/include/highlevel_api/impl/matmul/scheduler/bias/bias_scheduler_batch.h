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
* \file bias_scheduler_batch.h
* \brief
*/

#ifndef IMPL_MATMUL_SCHEDULER_BIAS_BIAS_SCHEDULER_BATCH_H
#define IMPL_MATMUL_SCHEDULER_BIAS_BIAS_SCHEDULER_BATCH_H

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
    !MatmulFeatureTrait<MM_CFG>::IsNeedUB() &&
    ToMatmulConfig(MM_CFG).enableSetBias &&
    A_TYPE::layout != LayoutMode::NONE &&
    ToMatmulConfig(MM_CFG).batchMode != BatchMode::SINGLE_LARGE_THAN_L1 &&
    (PhyPosIsUB(BIAS_TYPE::pos) || PhyPosIsGM(BIAS_TYPE::pos))>>
: public BiasSchedulerBase<IMPL, A_TYPE, B_TYPE, BIAS_TYPE, MM_CFG>
{
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(CopyBiasIn);
    MATMUL_USE_MODULE(LoadBias2C2);
    MATMUL_USE_MODULE(C1Buffer);
    MATMUL_USE_MODULE(C2Buffer);

    using BiasT = typename BIAS_TYPE::T;

public:
    using BASE_MODULE = AscendC::Impl::Detail::BiasSchedulerBase<IMPL,  A_TYPE,  B_TYPE,  BIAS_TYPE, MM_CFG>;

    __aicore__ inline BiasScheduler() = default;
    __aicore__ inline ~BiasScheduler() = default;

    __aicore__ inline void Init(int32_t batchNum = 0)
    {   auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        if (tiling.IsBias()) {
            MATMUL_MODULE(C1Buffer)->Init(batchNum * tiling.GetSingleCoreN());
            MATMUL_MODULE(C2Buffer)->Init();
        }
    }

    __aicore__ inline LocalTensor<BiasT> CopyIn(int32_t dataLen, int32_t dataNum = 1, int32_t srcOffset = 0)
    {
        LocalTensor<BiasT> biasC1;
        if (BASE_MODULE::enableBias_) {
            biasC1 = MATMUL_MODULE(C1Buffer)->AllocTensor();
            MATMUL_MODULE(CopyBiasIn)->Copy(biasC1, BASE_MODULE::srcTensor_, dataLen, dataNum, srcOffset);
            MATMUL_MODULE(C1Buffer)->EnQue(biasC1);
            MATMUL_MODULE(C1Buffer)->DeQue();
        }
        return biasC1;
    }

    __aicore__ inline void Free(LocalTensor<BiasT> &biasC1) {}

    __aicore__ inline void SplitLoad(LocalTensor<BiasT> &biasC1, int32_t dataLen = 0, int32_t srcOffset = 0)
    {
        if (BASE_MODULE::enableBias_ && MATMUL_MODULE(KLoop)->GetOuterIdx() == 0) {
            auto biasC2 = MATMUL_MODULE(C2Buffer)->Allocate();
            MATMUL_MODULE(LoadBias2C2)->Load(biasC2, biasC1[srcOffset], dataLen);
            MATMUL_MODULE(C2Buffer)->EnQue();
            MATMUL_MODULE(C2Buffer)->DeQue();
        }
    }

    __aicore__ inline void Free()
    {
        if (BASE_MODULE::enableBias_ &&
            MATMUL_MODULE(KLoop)->GetOuterIdx() == 0 &&
            MATMUL_MODULE(KLoop)->GetInnerIdx() == 0) {
            MATMUL_MODULE(C2Buffer)->Free();
        }
    }

    __aicore__ inline void End()
    {
        if (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().IsBias()) {
            MATMUL_MODULE(C1Buffer)->Destroy();
        }
    }

    __aicore__ inline void Destroy(LocalTensor<BiasT>& bias = NULL_TENSOR<BiasT>)
    {
        if (BASE_MODULE::enableBias_) {
            MATMUL_MODULE(C1Buffer)->FreeTensor(bias);
        }
        End();
    }

    __aicore__ inline void StopBias(LocalTensor<BiasT> &bias)
    {
        if (BASE_MODULE::enableBias_) {
            BASE_MODULE::SetBias(false);
            MATMUL_MODULE(C1Buffer)->FreeTensor(bias);
        }
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace Gemm
#endif // _BIAS_SCHEDULER_BATCH_H_
