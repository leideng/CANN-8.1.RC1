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
* \file load_bias_to_c2.h
* \brief load bias from c1 to c2 buffer
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_BIAS_LOAD_BIAS_TO_C2_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_BIAS_LOAD_BIAS_TO_C2_H

namespace AscendC {
namespace Impl {
namespace Detail {

/**
 * LoadBias2C2: responsible for load bias data into C2 buffer.
 * This module provides ablities to copy bias data in C2 Buffer.
 * We retain the freedom to make incompatible changes, but do not guarantee the stability.
 * LoadBias2C2 is only for internal usage, does not support extension or customized specialization!
 */
template <typename IMPL, class A_TYPE, class BIAS_TYPE, const auto &MM_CFG, typename = void>
class LoadBias2C2 {
    using L0cT = typename GetDstType<typename A_TYPE::T>::Type;
    using BiasT = typename BIAS_TYPE::T;

public:
    __aicore__ inline LoadBias2C2() = default;
    __aicore__ inline ~LoadBias2C2() = default;

    /**
     * @description: Load bias data from C1 to C2
     * @param: biasC2: dst tensor in c2
     * @param: bias: src tensor from l1
     * @param: dataLen: block length of one load
     * @return: void
     */
    __aicore__ inline void Load(const LocalTensor<L0cT>& biasC2, const LocalTensor<BiasT>& bias, int32_t dataLen) {}
};

template <typename IMPL, class A_TYPE, class BIAS_TYPE, const auto &MM_CFG>
class LoadBias2C2<IMPL, A_TYPE, BIAS_TYPE, MM_CFG, enable_if_t<
    ToMatmulConfig(MM_CFG).enableSetBias &&
    !MatmulFeatureTrait<MM_CFG>::IsNeedUB() &&
    (DoMatmulMDL(MM_CFG) || isNormEnableScheduler<A_TYPE, MM_CFG> ||
    IsBmmEnableScheduler<A_TYPE, MM_CFG> || DoMatmulSpecialMDL(MM_CFG) || IsBasicBlockEnable<MM_CFG> ||
    DoMatmulIBShareNorm(MM_CFG))>>
{
    using L0cT = typename GetDstType<typename A_TYPE::T>::Type;
    using BiasT = typename BIAS_TYPE::T;

public:
    __aicore__ inline LoadBias2C2() = default;
    __aicore__ inline ~LoadBias2C2() = default;

    __aicore__ inline void Load(const LocalTensor<L0cT>& biasC2, const LocalTensor<BiasT>& bias, int32_t dataLen)
    {
        constexpr auto biasType = IsSameType<L0cT, BiasT>::value ? 2 : 1; // 2:f32, 1:f16
        uint16_t lenBurst = (dataLen * biasType * 2 + 63) / 64;
        DataCopy(biasC2, bias, {1, lenBurst, 0, 0});
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _LOAD_BIAS_TO_C2_H_
