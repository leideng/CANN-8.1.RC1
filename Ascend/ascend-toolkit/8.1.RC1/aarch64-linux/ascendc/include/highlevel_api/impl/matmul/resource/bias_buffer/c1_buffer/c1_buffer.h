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
* \file c1_buffer.h
* \brief
*/

#ifndef IMPL_MATMUL_RESOURCE_BIAS_BUFFER_C1_BUFFER_C1_BUFFER_H
#define IMPL_MATMUL_RESOURCE_BIAS_BUFFER_C1_BUFFER_C1_BUFFER_H

#include "c1_buffer_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {

/**
 * C1Buffer: responsible for L1 bias buffer management.
 * This module provides ablities to allocate or free one l1 bias buffer block, and pipeline syncronization.
 * We retain the freedom to make incompatible changes, but do not guarantee the stability.
 * CopyBiasIn is only for internal usage, does not support extension or customized specialization!
 */
template <typename IMPL, class BIAS_TYPE, class A_TYPE, const auto& MM_CFG>
class C1Buffer<IMPL, BIAS_TYPE, A_TYPE, MM_CFG,
    enable_if_t<ToMatmulConfig(MM_CFG).enableSetBias &&
        (DoMatmulMDL(MM_CFG) || isNormEnableScheduler<A_TYPE, MM_CFG> ||
        IsBmmEnableScheduler<A_TYPE, MM_CFG> || DoMatmulSpecialMDL(MM_CFG) || IsBasicBlockEnable<MM_CFG> ||
        DoMatmulIBShareNorm(MM_CFG))>>
{
    using BiasT = typename BIAS_TYPE::T;
public:
    __aicore__ inline C1Buffer() {}
    __aicore__ inline ~C1Buffer() {}
    __aicore__ inline void Init(int32_t biasLen)
    {
        GetTPipePtr()->InitBuffer(qidBias_, 1, biasLen * sizeof(BiasT));
    }

    __aicore__ inline LocalTensor<BiasT> AllocTensor()
    {
        return qidBias_.template AllocTensor<BiasT>();
    }

    __aicore__ inline void FreeTensor(const LocalTensor<BiasT>& tensor = NULL_TENSOR<BiasT>)
    {
        qidBias_.FreeTensor(const_cast<LocalTensor<BiasT>&>(tensor));
    }

    __aicore__ inline void EnQue(LocalTensor<BiasT>& tensor)
    {
        qidBias_.EnQue(tensor);
    }

    __aicore__ inline LocalTensor<BiasT> DeQue()
    {
        return qidBias_.template DeQue<BiasT>();
    }

    __aicore__ inline void Destroy()
    {
        qidBias_.FreeAllEvent();
    }

private:
    TQue<TPosition::C1, QUEUE_DEPTH> qidBias_;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _C1_BUFFER_H_
