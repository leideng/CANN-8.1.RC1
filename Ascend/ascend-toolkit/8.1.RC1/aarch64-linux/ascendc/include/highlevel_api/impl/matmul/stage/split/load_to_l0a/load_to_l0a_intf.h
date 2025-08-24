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
 * \file load_to_l0a_intf.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_INTF_H
#define IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_INTF_H

#include "../load_to_l0_utils.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    LoadToL0A is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    LoadToL0A is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, typename A_TYPE, const auto& MM_CFG, typename = void>
class LoadToL0A
{
    using A_T = typename A_TYPE::T;
public:
    __aicore__ inline LoadToL0A() = default;
    __aicore__ inline ~LoadToL0A() = default;

    /**
     * @description: set scalar for gemv scalar mode
     * @param: scalar: init value
     * @return: void
     */
    __aicore__ inline void SetScalar(A_T scalar) {};

    /**
     * @description: Prepare for LoadToL0A
     * @param: isATranspose: A matrix transpose status
     * @param: aL1K: the length of K_axis for original aMatrix in L1
     * @param: aL1M: the length of M_axis for  original aMatrix in L1
     * @return: void
     */
    __aicore__ inline void Prepare(bool isATranspose, uint16_t aL1K, uint16_t aL1M) const {};

    /**
     * @description: load a base block from L1 to L0
     * @param: l0A: dst tensor in L0
     * @param: l1A: src tensor in L1
     * @param: aL1M: the length of M_axis for original aMatrix in L1
     * @param: aL1K: the length of K_axis for original aMatrix in L1
     * @param: madM: the length of M_axis for one base block
     * @param: madK: the length of K_axis for one base block
     * @param: aL1MOffset: Offset of the basic block relative to the original aMatrix in the m direction
     * @param: aL1KOffset: Offset of the basic block relative to the original aMatrix in the k direction
     * @param: isATranspose: A matrix transpose status
     * @return: void
     */
    __aicore__ inline void Load(const LocalTensor<A_T> &l0A, const LocalTensor<A_T> &l1A,
    uint16_t aL1M, uint16_t aL1K, uint16_t madM, uint16_t madK, uint16_t aL1MOffset, uint16_t aL1KOffset,
    bool isATranspose) const {};
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_INTF_H